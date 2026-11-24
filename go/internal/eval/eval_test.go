package eval

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/videoagent/videoagent_project/go/internal/agent"
	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

type fakeEvalReasoner struct{}

func (f *fakeEvalReasoner) ModelID() string { return "fake-eval" }
func (f *fakeEvalReasoner) Chat(_ context.Context, _ cloud.ChatRequest) (cloud.ChatResponse, error) {
	ans := "Mock evaluation answer."
	return cloud.ChatResponse{
		Message: cloud.ChatMessage{Role: "assistant", Content: &ans},
		Usage:   &cloud.TokenUsage{InputTokens: 40, OutputTokens: 10},
	}, nil
}

func TestAblationGroupsABCD(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "eval.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()

	if err := st.SaveSegments([]store.SegmentRecord{
		{VideoID: "v1", SegmentID: "seg_0000", StartMs: 0, EndMs: 30000, ASR: "hello", VisualStatus: "not_configured"},
	}); err != nil {
		t.Fatalf("save segment: %v", err)
	}

	r := &Runner{
		Reasoner:     &fakeEvalReasoner{},
		Store:        st,
		Tools:        &tools.VideoTools{VideoID: "v1", Store: st},
		WorkerConfig: agent.DefaultConfig(),
		MaxWorkers:   2,
	}
	questions := []QuestionItem{{ID: "q1", Question: "What is shown?"}}
	rep, err := r.Evaluate(context.Background(), "v1", questions, []Group{GroupA, GroupB, GroupC, GroupD})
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if len(rep.Results) != 4 || len(rep.Aggregates) != 4 {
		t.Fatalf("expected 4 group results and aggregates, got %d / %d", len(rep.Results), len(rep.Aggregates))
	}
	for _, res := range rep.Results {
		if res.Error != "" || res.Answer == "" {
			t.Fatalf("unexpected error/empty answer in group %s: %+v", res.Group, res)
		}
	}
}
