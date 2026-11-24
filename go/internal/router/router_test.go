package router

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/videoagent/videoagent_project/go/internal/agent"
	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

type mockRouterReasoner struct{}

func (m *mockRouterReasoner) ModelID() string { return "mock-router" }
func (m *mockRouterReasoner) Chat(_ context.Context, req cloud.ChatRequest) (cloud.ChatResponse, error) {
	sys := ""
	if len(req.Messages) > 0 && req.Messages[0].Content != nil {
		sys = *req.Messages[0].Content
	}
	if strings.Contains(sys, "Super-Agent Router for a long-video") {
		out := `{"tasks":[{"worker_id":"worker_asr","focus":"speech","question":"check speech"},{"worker_id":"worker_vis","focus":"vision","question":"check screen"}]}`
		return cloud.ChatResponse{Message: cloud.ChatMessage{Role: "assistant", Content: &out}}, nil
	}
	if strings.Contains(sys, "synthesizing final results") {
		out := "Synthesized answer combining worker_asr and worker_vis."
		return cloud.ChatResponse{Message: cloud.ChatMessage{Role: "assistant", Content: &out}}, nil
	}
	// Worker turn
	out := "Worker verified segment."
	return cloud.ChatResponse{
		Message: cloud.ChatMessage{Role: "assistant", Content: &out},
		Usage:   &cloud.TokenUsage{InputTokens: 50, OutputTokens: 10},
	}, nil
}

func TestRouterConcurrentWorkers(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "router.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()

	if err := st.SaveSegments([]store.SegmentRecord{
		{VideoID: "v1", SegmentID: "seg_0000", StartMs: 0, EndMs: 30000, ASR: "intro", VisualStatus: "not_configured"},
	}); err != nil {
		t.Fatalf("save segments: %v", err)
	}

	r := &Router{
		Reasoner:     &mockRouterReasoner{},
		Store:        st,
		Tools:        &tools.VideoTools{VideoID: "v1", Store: st},
		WorkerConfig: agent.DefaultConfig(),
		MaxWorkers:   2,
	}
	res, err := r.Ask(context.Background(), "v1", "What happened in the intro?")
	if err != nil {
		t.Fatalf("router ask: %v", err)
	}
	if len(res.WorkerResults) != 2 {
		t.Fatalf("expected 2 worker results, got %d", len(res.WorkerResults))
	}
	if !strings.Contains(res.FinalAnswer, "Synthesized answer") {
		t.Fatalf("unexpected final answer: %s", res.FinalAnswer)
	}
	if res.TotalStepsUsed != 2 {
		t.Fatalf("expected 2 total worker steps, got %d", res.TotalStepsUsed)
	}
}
