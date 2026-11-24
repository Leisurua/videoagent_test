package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/videoagent/videoagent_project/go/internal/agent"
	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/rag"
	"github.com/videoagent/videoagent_project/go/internal/router"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

type Group string

const (
	GroupA Group = "A" // Plain text RAG -> single LLM answer
	GroupB Group = "B" // Text RAG + ReAct (no raw visual tools)
	GroupC Group = "C" // Text RAG + ReAct + raw video observation
	GroupD Group = "D" // Dual-channel RAG + Super-Agent Router + ReAct + raw video observation
)

type QuestionItem struct {
	ID              string `json:"id"`
	Question        string `json:"question"`
	ReferenceAnswer string `json:"reference_answer,omitempty"`
	EvidenceStartMs int64  `json:"evidence_start_ms,omitempty"`
	EvidenceEndMs   int64  `json:"evidence_end_ms,omitempty"`
	Category        string `json:"category,omitempty"`
}

type RunMetrics struct {
	QuestionID        string   `json:"question_id"`
	Group             Group    `json:"group"`
	Answer            string   `json:"answer"`
	TerminationReason string   `json:"termination_reason"`
	LatencyMs         int64    `json:"latency_ms"`
	StepsUsed         int      `json:"steps_used"`
	ModelCalls        int      `json:"model_calls"`
	ToolCalls         int      `json:"tool_calls"`
	InputTokens       int      `json:"input_tokens"`
	OutputTokens      int      `json:"output_tokens"`
	EvidenceCount     int      `json:"evidence_count"`
	EvidenceIDs       []string `json:"evidence_ids,omitempty"`
	Error             string   `json:"error,omitempty"`
}

type GroupAggregate struct {
	Group           Group   `json:"group"`
	Runs            int     `json:"runs"`
	MeanLatencyMs   float64 `json:"mean_latency_ms"`
	MeanStepsUsed   float64 `json:"mean_steps_used"`
	TotalTokens     int     `json:"total_tokens"`
	TotalEvidence   int     `json:"total_evidence"`
}

type Report struct {
	VideoID    string           `json:"video_id"`
	Groups     []Group          `json:"groups"`
	Results    []RunMetrics     `json:"results"`
	Aggregates []GroupAggregate `json:"aggregates"`
}

type Runner struct {
	Reasoner     cloud.ChatReasoner
	Store        *store.Store
	Tools        *tools.VideoTools
	TextOnlyRAG  *rag.Engine
	DualRAG      *rag.Engine
	WorkerConfig agent.ExecutionConfig
	MaxWorkers   int
}

func (r *Runner) Evaluate(ctx context.Context, videoID string, questions []QuestionItem, groups []Group) (*Report, error) {
	if len(groups) == 0 {
		groups = []Group{GroupA, GroupB, GroupC, GroupD}
	}
	report := &Report{
		VideoID: videoID,
		Groups:  groups,
	}

	for _, g := range groups {
		var sumLat, sumSteps int64
		var sumTok, sumEv, count int
		for _, q := range questions {
			m := r.runOne(ctx, videoID, q, g)
			report.Results = append(report.Results, m)
			count++
			sumLat += m.LatencyMs
			sumSteps += int64(m.StepsUsed)
			sumTok += m.InputTokens + m.OutputTokens
			sumEv += m.EvidenceCount
		}
		agg := GroupAggregate{
			Group:         g,
			Runs:          count,
			TotalTokens:   sumTok,
			TotalEvidence: sumEv,
		}
		if count > 0 {
			agg.MeanLatencyMs = float64(sumLat) / float64(count)
			agg.MeanStepsUsed = float64(sumSteps) / float64(count)
		}
		report.Aggregates = append(report.Aggregates, agg)
	}
	return report, nil
}

func (r *Runner) runOne(ctx context.Context, videoID string, q QuestionItem, g Group) RunMetrics {
	started := time.Now()
	m := RunMetrics{QuestionID: q.ID, Group: g}

	switch g {
	case GroupA:
		ragEng := r.textRAG()
		var ragJSON string
		if ragEng != nil {
			res, err := ragEng.Search(ctx, videoID, q.Question, 5)
			if err == nil {
				b, _ := json.Marshal(res)
				ragJSON = string(b)
			}
		}
		sys := "Answer the question using only the retrieved video text segments. Cite segment timestamps when available."
		usr := fmt.Sprintf("Question: %s\n\nRetrieved segments:\n%s", q.Question, ragJSON)
		resp, err := r.Reasoner.Chat(ctx, cloud.ChatRequest{
			Messages: []cloud.ChatMessage{
				{Role: "system", Content: &sys},
				{Role: "user", Content: &usr},
			},
		})
		m.LatencyMs = time.Since(started).Milliseconds()
		if err != nil {
			m.TerminationReason = "model_api_error"
			m.Error = err.Error()
			return m
		}
		m.StepsUsed = 1
		m.ModelCalls = 1
		m.TerminationReason = "completed"
		if resp.Message.Content != nil {
			m.Answer = *resp.Message.Content
		}
		if resp.Usage != nil {
			m.InputTokens = resp.Usage.InputTokens
			m.OutputTokens = resp.Usage.OutputTokens
		}
		return m

	case GroupB, GroupC:
		cfg := r.WorkerConfig
		if cfg.ExecutionLimit <= 0 {
			cfg = agent.DefaultConfig()
		}
		if g == GroupB {
			cfg.DisabledTools = map[string]bool{
				"inspect_segment": true,
				"locate_event":    true,
				"get_frame":       true,
				"inspect_frame":   true,
				"ocr_frame":       true,
				"crop_frame":      true,
			}
		}
		var pre *rag.SearchResult
		if ragEng := r.textRAG(); ragEng != nil {
			if res, err := ragEng.Search(ctx, videoID, q.Question, 5); err == nil {
				pre = &res
			}
		}
		runner := &agent.AgentRunner{
			Reasoner: r.Reasoner,
			Tools:    r.Tools,
			Store:    r.Store,
			Config:   cfg,
		}
		state, err := runner.Run(ctx, q.Question, videoID, "eval_"+string(g), pre)
		m.LatencyMs = time.Since(started).Milliseconds()
		if err != nil {
			m.TerminationReason = "error"
			m.Error = err.Error()
			return m
		}
		m.Answer = state.Answer
		m.TerminationReason = state.TerminationReason
		m.StepsUsed = state.Budget.StepsUsed
		m.ModelCalls = state.Budget.ModelCalls
		m.ToolCalls = state.Budget.ToolCalls
		m.InputTokens = state.Budget.InputTokens
		m.OutputTokens = state.Budget.OutputTokens
		m.EvidenceIDs = state.EvidenceIDs
		m.EvidenceCount = len(state.EvidenceIDs)
		return m

	case GroupD:
		vt := *r.Tools
		if r.DualRAG != nil {
			vt.RAG = r.DualRAG
		}
		rt := &router.Router{
			Reasoner:     r.Reasoner,
			Store:        r.Store,
			Tools:        &vt,
			WorkerConfig: r.WorkerConfig,
			MaxWorkers:   r.MaxWorkers,
		}
		res, err := rt.Ask(ctx, videoID, q.Question)
		m.LatencyMs = time.Since(started).Milliseconds()
		if err != nil {
			m.TerminationReason = "error"
			m.Error = err.Error()
			return m
		}
		m.Answer = res.FinalAnswer
		m.TerminationReason = "completed"
		m.StepsUsed = res.TotalStepsUsed
		m.EvidenceCount = len(res.Evidence)
		for _, w := range res.WorkerResults {
			m.ModelCalls += w.Budget.ModelCalls
			m.ToolCalls += w.Budget.ToolCalls
			m.InputTokens += w.Budget.InputTokens
			m.OutputTokens += w.Budget.OutputTokens
			m.EvidenceIDs = append(m.EvidenceIDs, w.EvidenceIDs...)
		}
		return m

	default:
		m.Error = fmt.Sprintf("unknown group %q", g)
		return m
	}
}

func (r *Runner) textRAG() *rag.Engine {
	if r.TextOnlyRAG != nil {
		return r.TextOnlyRAG
	}
	if r.Tools != nil {
		return r.Tools.RAG
	}
	return nil
}
