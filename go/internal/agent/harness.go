package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/rag"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

// ExecutionConfig configures ReAct budget and safety bounds.
type ExecutionConfig struct {
	ExecutionLimit          int
	ForceAnswerRemaining    int
	WarningThresholds       []float64
	MaxConsecutiveToolCalls int
	MaxConsecutiveFailures  int
	MaxResultChars          int
	DisabledTools           map[string]bool
}

func DefaultConfig() ExecutionConfig {
	return ExecutionConfig{
		ExecutionLimit:          125,
		ForceAnswerRemaining:    4,
		WarningThresholds:       []float64{0.65, 0.80, 0.90},
		MaxConsecutiveToolCalls: 5,
		MaxConsecutiveFailures:  3,
		MaxResultChars:          6000,
	}
}

// BudgetState tracks execution steps (1 step = 1 model call or 1 actual tool call).
type BudgetState struct {
	StepsUsed           int `json:"steps_used"`
	ModelCalls          int `json:"model_calls"`
	ToolCalls           int `json:"tool_calls"`
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	UsageReportedCalls  int `json:"usage_reported_calls"`
	ConsecutiveFailures int `json:"consecutive_failures"`
}

// ExecutionGuard enforces budget, deduplication, and failure wrap-up.
type ExecutionGuard struct {
	cfg          ExecutionConfig
	budget       BudgetState
	consecTools  int
	warningsSent map[float64]bool
	seenCalls    map[string]bool
}

func NewExecutionGuard(cfg ExecutionConfig) *ExecutionGuard {
	if cfg.ExecutionLimit <= 0 {
		cfg = DefaultConfig()
	}
	return &ExecutionGuard{
		cfg:          cfg,
		warningsSent: make(map[float64]bool),
		seenCalls:    make(map[string]bool),
	}
}

func (g *ExecutionGuard) Budget() BudgetState { return g.budget }

func (g *ExecutionGuard) ShouldForceAnswer() (bool, string) {
	rem := g.cfg.ExecutionLimit - g.budget.StepsUsed
	if rem <= g.cfg.ForceAnswerRemaining {
		return true, "budget_exhausted"
	}
	if g.cfg.MaxConsecutiveFailures > 0 && g.budget.ConsecutiveFailures >= g.cfg.MaxConsecutiveFailures {
		return true, "consecutive_tool_failures"
	}
	if g.cfg.MaxConsecutiveToolCalls > 0 && g.consecTools >= g.cfg.MaxConsecutiveToolCalls {
		return true, "consecutive_tool_calls"
	}
	return false, ""
}

func (g *ExecutionGuard) ThresholdWarning() string {
	ratio := float64(g.budget.StepsUsed) / float64(g.cfg.ExecutionLimit)
	for _, t := range g.cfg.WarningThresholds {
		if ratio >= t && !g.warningsSent[t] {
			g.warningsSent[t] = true
			return fmt.Sprintf("Budget notice: %.0f%% of %d execution steps used (%d remaining). Converge toward a verified answer.",
				t*100, g.cfg.ExecutionLimit, g.cfg.ExecutionLimit-g.budget.StepsUsed)
		}
	}
	return ""
}

func (g *ExecutionGuard) CheckToolCall(name, argsJSON string) (bool, string) {
	if g.cfg.DisabledTools[name] {
		return false, fmt.Sprintf("tool %q is disabled", name)
	}
	if force, reason := g.ShouldForceAnswer(); force {
		return false, fmt.Sprintf("execution guard triggered (%s); no more tool calls allowed", reason)
	}
	key := name + "::" + canonicalJSON(argsJSON)
	if g.seenCalls[key] {
		return false, fmt.Sprintf("duplicate call to %s with identical arguments blocked", name)
	}
	g.seenCalls[key] = true
	return true, ""
}

func (g *ExecutionGuard) RecordModelCall(usage *cloud.TokenUsage) {
	g.budget.StepsUsed++
	g.budget.ModelCalls++
	g.consecTools = 0
	if usage != nil && (usage.InputTokens > 0 || usage.OutputTokens > 0) {
		g.budget.InputTokens += usage.InputTokens
		g.budget.OutputTokens += usage.OutputTokens
		g.budget.UsageReportedCalls++
	}
}

func (g *ExecutionGuard) RecordToolResult(ok bool) {
	g.budget.StepsUsed++
	g.budget.ToolCalls++
	g.consecTools++
	if ok {
		g.budget.ConsecutiveFailures = 0
	} else {
		g.budget.ConsecutiveFailures++
	}
}

// AgentState is the complete execution trace of a Worker run.
type AgentState struct {
	RunID             string               `json:"run_id"`
	WorkerID          string               `json:"worker_id"`
	VideoID           string               `json:"video_id"`
	Question          string               `json:"question"`
	Answer            string               `json:"answer"`
	TerminationReason string               `json:"termination_reason"`
	Budget            BudgetState          `json:"budget_state"`
	InitialRAG        *rag.SearchResult    `json:"initial_rag,omitempty"`
	Messages          []cloud.ChatMessage  `json:"messages"`
	EvidenceIDs       []string             `json:"evidence_ids"`
}

// AgentRunner executes one isolated Worker ReAct loop.
type AgentRunner struct {
	Reasoner cloud.ChatReasoner
	Tools    *tools.VideoTools
	Store    *store.Store
	Config   ExecutionConfig
}

const workerSystemPrompt = `You are a Video Analysis Worker operating inside a ReAct Harness.
Use dual-channel RAG segments only as unverified candidate locations.
Always verify visual details by calling inspect_segment, inspect_frame, locate_event, get_frame, crop_frame, or ocr_frame on the original video.
Cite concrete timestamps (ms or mm:ss) and frame refs in your final answer. Do not invent unseen facts.`

func (r *AgentRunner) Run(ctx context.Context, question, videoID, workerID string, precomputed *rag.SearchResult) (*AgentState, error) {
	if workerID == "" {
		workerID = "worker_1"
	}
	runID := newID("run")
	guard := NewExecutionGuard(r.Config)
	if r.Config.MaxResultChars <= 0 {
		r.Config.MaxResultChars = 6000
	}

	// Initial RAG (does not consume loop budget).
	var initialRAG *rag.SearchResult
	if precomputed != nil {
		initialRAG = precomputed
	} else if r.Tools != nil && r.Tools.RAG != nil {
		res, err := r.Tools.RAG.Search(ctx, videoID, question, 5)
		if err == nil {
			initialRAG = &res
		}
	}

	userMsg := fmt.Sprintf("Video ID: %s\nQuestion: %s", videoID, question)
	if initialRAG != nil && len(initialRAG.Segments) > 0 {
		raw, _ := json.Marshal(initialRAG)
		userMsg += "\n\nInitial RAG candidates (unverified):\n" + string(raw)
	}

	messages := []cloud.ChatMessage{
		{Role: "system", Content: strPtr(workerSystemPrompt)},
		{Role: "user", Content: strPtr(userMsg)},
	}

	toolDefs := r.Tools.Definitions()
	toolDefs = append(toolDefs, readToolResultDef())
	fullResults := make(map[string]string)
	var evidenceIDs []string

	state := &AgentState{
		RunID:      runID,
		WorkerID:   workerID,
		VideoID:    videoID,
		Question:   question,
		InitialRAG: initialRAG,
	}

	for {
		force, reason := guard.ShouldForceAnswer()
		reqTools := toolDefs
		toolChoice := "auto"
		if force {
			reqTools = nil
			toolChoice = "none"
			messages = append(messages, cloud.ChatMessage{
				Role:    "user",
				Content: strPtr(fmt.Sprintf("Execution limit reached (%s). Provide your final answer now using only verified evidence already collected.", reason)),
			})
		} else if warn := guard.ThresholdWarning(); warn != "" {
			messages = append(messages, cloud.ChatMessage{Role: "user", Content: strPtr(warn)})
		}

		resp, err := r.Reasoner.Chat(ctx, cloud.ChatRequest{
			Messages:   messages,
			Tools:      reqTools,
			ToolChoice: toolChoice,
		})
		if err != nil {
			state.TerminationReason = "model_api_error"
			state.Answer = err.Error()
			break
		}
		guard.RecordModelCall(resp.Usage)
		messages = append(messages, resp.Message)

		if len(resp.Message.ToolCalls) == 0 || force {
			if resp.Message.Content != nil && strings.TrimSpace(*resp.Message.Content) != "" {
				state.Answer = *resp.Message.Content
				if force {
					state.TerminationReason = reason
				} else {
					state.TerminationReason = "completed"
				}
			} else {
				state.TerminationReason = "no_final_answer"
			}
			break
		}

		for _, tc := range resp.Message.ToolCalls {
			allowed, denyReason := guard.CheckToolCall(tc.Function.Name, tc.Function.Arguments)
			var payloadStr string
			if !allowed {
				b, _ := json.Marshal(map[string]any{"status": "blocked", "reason": denyReason})
				payloadStr = string(b)
			} else if tc.Function.Name == "read_tool_result" {
				payloadStr = r.handleReadResult(tc.Function.Arguments, fullResults, guard)
			} else {
				resMap, execErr := r.Tools.Execute(ctx, tc.Function.Name, tc.Function.Arguments)
				if execErr != nil {
					guard.RecordToolResult(false)
					b, _ := json.Marshal(map[string]any{"status": "error", "error": execErr.Error()})
					payloadStr = string(b)
				} else {
					guard.RecordToolResult(true)
					if evID := r.maybeRecordEvidence(runID, workerID, videoID, tc.Function.Name, resMap); evID != "" {
						evidenceIDs = append(evidenceIDs, evID)
					}
					b, _ := json.Marshal(resMap)
					payloadStr = string(b)
				}
			}

			if len(payloadStr) > r.Config.MaxResultChars {
				refID := fmt.Sprintf("res_%d", len(fullResults)+1)
				fullResults[refID] = payloadStr
				preview, _ := json.Marshal(map[string]any{
					"truncated":  true,
					"result_ref": refID,
					"total_len":  len(payloadStr),
					"preview":    payloadStr[:r.Config.MaxResultChars],
				})
				payloadStr = string(preview)
			}

			messages = append(messages, cloud.ChatMessage{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    strPtr(payloadStr),
			})
		}
	}

	state.Budget = guard.Budget()
	state.Messages = messages
	state.EvidenceIDs = evidenceIDs
	return state, nil
}

func (r *AgentRunner) handleReadResult(argsJSON string, full map[string]string, guard *ExecutionGuard) string {
	var a struct {
		ResultRef string `json:"result_ref"`
		Offset    int    `json:"offset"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
		guard.RecordToolResult(false)
		return `{"status":"error","error":"invalid arguments"}`
	}
	raw, ok := full[a.ResultRef]
	if !ok {
		guard.RecordToolResult(false)
		return `{"status":"error","error":"result_ref not found"}`
	}
	guard.RecordToolResult(true)
	if a.Offset < 0 || a.Offset >= len(raw) {
		return `{"content":"","eof":true}`
	}
	end := a.Offset + r.Config.MaxResultChars
	if end > len(raw) {
		end = len(raw)
	}
	b, _ := json.Marshal(map[string]any{
		"content": raw[a.Offset:end],
		"offset":  a.Offset,
		"eof":     end == len(raw),
	})
	return string(b)
}

func (r *AgentRunner) maybeRecordEvidence(runID, workerID, videoID, toolName string, res map[string]any) string {
	if r.Store == nil {
		return ""
	}
	switch toolName {
	case "inspect_segment", "inspect_frame", "locate_event":
		status, _ := res["status"].(string)
		if status != "matched" && status != "found" {
			return ""
		}
		ans, _ := res["answer"].(string)
		startMs := toInt64(res["start_ms"])
		endMs := toInt64(res["end_ms"])
		var refs []string
		if rawRefs, ok := res["evidence_refs"].([]any); ok {
			for _, x := range rawRefs {
				if s, ok := x.(string); ok {
					refs = append(refs, s)
				}
			}
		}
		evID := newID("ev")
		_ = r.Store.AddEvidence(store.EvidenceRecord{
			EvidenceID:  evID,
			RunID:       runID,
			WorkerID:    workerID,
			VideoID:     videoID,
			StartMs:     startMs,
			EndMs:       endMs,
			Source:      toolName,
			Observation: ans,
			Refs:        refs,
			Verified:    true,
			CreatedAt:   time.Now().UnixMilli(),
		})
		return evID
	}
	return ""
}

func readToolResultDef() cloud.ToolDefinition {
	var td cloud.ToolDefinition
	td.Type = "function"
	td.Function.Name = "read_tool_result"
	td.Function.Description = "Read a truncated tool result page by result_ref and character offset."
	td.Function.Parameters = map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result_ref": map[string]any{"type": "string"},
			"offset":     map[string]any{"type": "integer"},
		},
		"required": []string{"result_ref"},
	}
	return td
}

func canonicalJSON(s string) string {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return strings.TrimSpace(s)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return strings.TrimSpace(s)
	}
	return string(b)
}

func toInt64(v any) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int64:
		return n
	case int:
		return int64(n)
	}
	return 0
}

func strPtr(s string) *string { return &s }

func newID(prefix string) string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}
