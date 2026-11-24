package router

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/videoagent/videoagent_project/go/internal/agent"
	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/rag"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

// WorkerTask defines one focused sub-investigation spawned by the Super-Agent Router.
type WorkerTask struct {
	WorkerID string `json:"worker_id"`
	Focus    string `json:"focus"` // e.g. "asr_context", "visual_code_verification"
	Question string `json:"question"`
}

// WorkerSummary is the distilled output returned by a Worker to the Router.
// Raw tool traces stay isolated inside each Worker; Router only sees summaries + evidence IDs.
type WorkerSummary struct {
	WorkerID          string            `json:"worker_id"`
	RunID             string            `json:"run_id"`
	Focus             string            `json:"focus"`
	Answer            string            `json:"answer"`
	TerminationReason string            `json:"termination_reason"`
	Budget            agent.BudgetState `json:"budget"`
	EvidenceIDs       []string          `json:"evidence_ids"`
	Error             string            `json:"error,omitempty"`
}

// RouterResult is the final aggregated response produced by the Super-Agent Router.
type RouterResult struct {
	VideoID        string                 `json:"video_id"`
	Question       string                 `json:"question"`
	FinalAnswer    string                 `json:"final_answer"`
	Tasks          []WorkerTask           `json:"tasks"`
	WorkerResults  []WorkerSummary        `json:"worker_results"`
	Evidence       []store.EvidenceRecord `json:"evidence"`
	TotalStepsUsed int                    `json:"total_steps_used"`
}

// Router is the Super-Agent Global Coordinator.
// It never performs fine-grained video observation itself; it plans sub-tasks,
// dispatches isolated Worker Sessions concurrently, and synthesizes Evidence Store records.
type Router struct {
	Reasoner     cloud.ChatReasoner
	Store        *store.Store
	Tools        *tools.VideoTools
	WorkerConfig agent.ExecutionConfig
	MaxWorkers   int
}

const plannerPrompt = `You are the Super-Agent Router for a long-video understanding system.
Given a user question and initial unverified RAG candidate segments, decide whether to dispatch 1 or up to %d independent Worker agents concurrently (e.g. one for speech/concept flow, one for visual screen/code verification).
Return ONLY a JSON object: {"tasks": [{"worker_id": "worker_1", "focus": "...", "question": "..."}]}.`

const synthesisPrompt = `You are the Super-Agent Router synthesizing final results.
Combine the Worker summaries and verified Evidence Store records into a clear, accurate final answer.
Cite exact timestamps and frame refs from the verified evidence. If Workers disagree or evidence is missing, state the uncertainty explicitly.`

func (r *Router) Ask(ctx context.Context, videoID, question string) (*RouterResult, error) {
	maxWorkers := r.MaxWorkers
	if maxWorkers <= 0 {
		maxWorkers = 2
	}

	// 1. Run initial dual-channel RAG once at the Router level to share candidates.
	var initialRAG *rag.SearchResult
	if r.Tools != nil && r.Tools.RAG != nil {
		res, err := r.Tools.RAG.Search(ctx, videoID, question, 5)
		if err == nil {
			initialRAG = &res
		}
	}

	// 2. Plan sub-tasks for independent Workers.
	tasks := r.planTasks(ctx, videoID, question, initialRAG, maxWorkers)

	// 3. Dispatch Workers concurrently with bounded per-worker budget.
	workerCfg := r.WorkerConfig
	if workerCfg.ExecutionLimit <= 0 {
		workerCfg = agent.DefaultConfig()
	}
	if len(tasks) > 1 {
		// Split total step budget across concurrent workers.
		perWorker := workerCfg.ExecutionLimit / len(tasks)
		if perWorker < 10 {
			perWorker = 10
		}
		workerCfg.ExecutionLimit = perWorker
	}

	summaries := make([]WorkerSummary, len(tasks))
	var wg sync.WaitGroup
	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, t WorkerTask) {
			defer wg.Done()
			runner := &agent.AgentRunner{
				Reasoner: r.Reasoner,
				Tools:    r.Tools,
				Store:    r.Store,
				Config:   workerCfg,
			}
			state, err := runner.Run(ctx, t.Question, videoID, t.WorkerID, initialRAG)
			if err != nil {
				summaries[idx] = WorkerSummary{
					WorkerID: t.WorkerID,
					Focus:    t.Focus,
					Error:    err.Error(),
				}
				return
			}
			summaries[idx] = WorkerSummary{
				WorkerID:          t.WorkerID,
				RunID:             state.RunID,
				Focus:             t.Focus,
				Answer:            state.Answer,
				TerminationReason: state.TerminationReason,
				Budget:            state.Budget,
				EvidenceIDs:       state.EvidenceIDs,
			}
		}(i, task)
	}
	wg.Wait()

	// 4. Collect all verified Evidence Store records across Workers.
	var allEvidence []store.EvidenceRecord
	totalSteps := 0
	for _, s := range summaries {
		totalSteps += s.Budget.StepsUsed
		if r.Store != nil && s.RunID != "" {
			evs, err := r.Store.ListEvidence(s.RunID, videoID)
			if err == nil {
				allEvidence = append(allEvidence, evs...)
			}
		}
	}

	// 5. Synthesize final answer.
	finalAnswer := r.synthesize(ctx, question, summaries, allEvidence)

	return &RouterResult{
		VideoID:        videoID,
		Question:       question,
		FinalAnswer:    finalAnswer,
		Tasks:          tasks,
		WorkerResults:  summaries,
		Evidence:       allEvidence,
		TotalStepsUsed: totalSteps,
	}, nil
}

func (r *Router) planTasks(ctx context.Context, videoID, question string, initialRAG *rag.SearchResult, maxWorkers int) []WorkerTask {
	fallback := []WorkerTask{{
		WorkerID: "worker_1",
		Focus:    "general_investigation",
		Question: question,
	}}
	if r.Reasoner == nil {
		return fallback
	}

	ragSnippet := ""
	if initialRAG != nil && len(initialRAG.Segments) > 0 {
		b, _ := json.Marshal(initialRAG.Segments)
		ragSnippet = "\nCandidate segments:\n" + string(b)
	}
	sys := fmt.Sprintf(plannerPrompt, maxWorkers)
	usr := fmt.Sprintf("Video ID: %s\nQuestion: %s%s", videoID, question, ragSnippet)

	resp, err := r.Reasoner.Chat(ctx, cloud.ChatRequest{
		Messages: []cloud.ChatMessage{
			{Role: "system", Content: &sys},
			{Role: "user", Content: &usr},
		},
	})
	if err != nil || resp.Message.Content == nil {
		return fallback
	}

	raw := extractJSON(*resp.Message.Content)
	var parsed struct {
		Tasks []WorkerTask `json:"tasks"`
	}
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil || len(parsed.Tasks) == 0 {
		return fallback
	}
	if len(parsed.Tasks) > maxWorkers {
		parsed.Tasks = parsed.Tasks[:maxWorkers]
	}
	for i := range parsed.Tasks {
		if strings.TrimSpace(parsed.Tasks[i].WorkerID) == "" {
			parsed.Tasks[i].WorkerID = fmt.Sprintf("worker_%d", i+1)
		}
		if strings.TrimSpace(parsed.Tasks[i].Question) == "" {
			parsed.Tasks[i].Question = question
		}
	}
	return parsed.Tasks
}

func (r *Router) synthesize(ctx context.Context, question string, summaries []WorkerSummary, evidence []store.EvidenceRecord) string {
	if len(summaries) == 1 && summaries[0].Error == "" && summaries[0].Answer != "" {
		return summaries[0].Answer
	}
	if r.Reasoner == nil {
		var parts []string
		for _, s := range summaries {
			if s.Answer != "" {
				parts = append(parts, fmt.Sprintf("[%s] %s", s.WorkerID, s.Answer))
			}
		}
		return strings.Join(parts, "\n\n")
	}

	payload, _ := json.Marshal(map[string]any{
		"question":       question,
		"worker_results": summaries,
		"evidence_store": evidence,
	})
	sys := synthesisPrompt
	usr := string(payload)
	resp, err := r.Reasoner.Chat(ctx, cloud.ChatRequest{
		Messages: []cloud.ChatMessage{
			{Role: "system", Content: &sys},
			{Role: "user", Content: &usr},
		},
	})
	if err != nil || resp.Message.Content == nil || strings.TrimSpace(*resp.Message.Content) == "" {
		var parts []string
		for _, s := range summaries {
			if s.Answer != "" {
				parts = append(parts, fmt.Sprintf("[%s] %s", s.WorkerID, s.Answer))
			}
		}
		return strings.Join(parts, "\n\n")
	}
	return *resp.Message.Content
}

func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	if start := strings.Index(s, "{"); start >= 0 {
		if end := strings.LastIndex(s, "}"); end > start {
			return s[start : end+1]
		}
	}
	return s
}
