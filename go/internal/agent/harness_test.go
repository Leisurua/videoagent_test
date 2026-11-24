package agent

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/rag"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

type fakeReasoner struct {
	step int
}

func (f *fakeReasoner) ModelID() string { return "fake-reasoner" }
func (f *fakeReasoner) Chat(_ context.Context, req cloud.ChatRequest) (cloud.ChatResponse, error) {
	f.step++
	if f.step == 1 {
		var tc cloud.ToolCall
		tc.ID = "call_1"
		tc.Type = "function"
		tc.Function.Name = "get_segment"
		tc.Function.Arguments = `{"segment_id":"seg_0000"}`
		return cloud.ChatResponse{
			Message: cloud.ChatMessage{Role: "assistant", ToolCalls: []cloud.ToolCall{tc}},
			Usage:   &cloud.TokenUsage{InputTokens: 100, OutputTokens: 20},
		}, nil
	}
	ans := "Verified at 0-30000ms."
	return cloud.ChatResponse{
		Message: cloud.ChatMessage{Role: "assistant", Content: &ans},
		Usage:   &cloud.TokenUsage{InputTokens: 120, OutputTokens: 30},
	}, nil
}

func TestExecutionGuardDeduplicationAndBudget(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ExecutionLimit = 10
	cfg.ForceAnswerRemaining = 2
	g := NewExecutionGuard(cfg)

	ok, _ := g.CheckToolCall("get_segment", `{"segment_id": "seg_0000"}`)
	if !ok {
		t.Fatalf("first call should be allowed")
	}
	ok, reason := g.CheckToolCall("get_segment", `{"segment_id":"seg_0000"}`)
	if ok || reason == "" {
		t.Fatalf("duplicate call with same canonical JSON should be blocked")
	}
}

func TestAgentRunnerLoop(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "agent.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()

	if err := st.SaveSegments([]store.SegmentRecord{
		{VideoID: "v1", SegmentID: "seg_0000", StartMs: 0, EndMs: 30000, ASR: "hello world", VisualStatus: "not_configured"},
	}); err != nil {
		t.Fatalf("save segment: %v", err)
	}

	runner := &AgentRunner{
		Reasoner: &fakeReasoner{},
		Tools:    &tools.VideoTools{VideoID: "v1", Store: st},
		Store:    st,
		Config:   DefaultConfig(),
	}
	pre := &rag.SearchResult{Status: "ok", Segments: []rag.SearchHit{{SegmentID: "seg_0000", StartMs: 0, EndMs: 30000}}}
	state, err := runner.Run(context.Background(), "what was said?", "v1", "worker_a", pre)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if state.TerminationReason != "completed" {
		t.Fatalf("expected completed, got %s", state.TerminationReason)
	}
	if state.Budget.ModelCalls != 2 || state.Budget.ToolCalls != 1 || state.Budget.StepsUsed != 3 {
		t.Fatalf("unexpected budget: %+v", state.Budget)
	}
}
