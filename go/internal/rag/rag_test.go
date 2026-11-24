package rag

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/videoagent/videoagent_project/go/internal/store"
)

type fakeTextEmbedder struct{}

func (f *fakeTextEmbedder) ModelID() string { return "fake-text" }
func (f *fakeTextEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if t == "error trace" {
			out[i] = []float32{1, 0, 0}
		} else {
			out[i] = []float32{0, 1, 0}
		}
	}
	return out, nil
}

type fakeVisualEmbedder struct{}

func (f *fakeVisualEmbedder) ModelID() string { return "fake-visual" }
func (f *fakeVisualEmbedder) EmbedImages(_ context.Context, paths []string) ([][]float32, error) {
	return make([][]float32, len(paths)), nil
}
func (f *fakeVisualEmbedder) EmbedQueries(_ context.Context, queries []string) ([][]float32, error) {
	out := make([][]float32, len(queries))
	for i := range queries {
		out[i] = []float32{1, 0, 0}
	}
	return out, nil
}

func TestDualChannelRRFFusion(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()

	err = st.SaveSegments([]store.SegmentRecord{
		{VideoID: "v1", SegmentID: "seg_0000", StartMs: 0, EndMs: 30000, ASR: "intro", VisualStatus: "not_configured"},
		{VideoID: "v1", SegmentID: "seg_0001", StartMs: 30000, EndMs: 60000, ASR: "silent code", VisualStatus: "not_configured"},
	})
	if err != nil {
		t.Fatalf("save segments: %v", err)
	}

	err = st.SaveVectors([]store.VectorRecord{
		{VideoID: "v1", Channel: "text", SegmentID: "seg_0000", ModelID: "fake-text", Vector: []float32{1, 0, 0}},
		{VideoID: "v1", Channel: "text", SegmentID: "seg_0001", ModelID: "fake-text", Vector: []float32{0, 1, 0}},
		{VideoID: "v1", Channel: "visual", SegmentID: "seg_0001", ModelID: "fake-visual", Vector: []float32{1, 0, 0}},
		{VideoID: "v1", Channel: "visual", SegmentID: "seg_0000", ModelID: "fake-visual", Vector: []float32{0.5, 0.5, 0}},
	})
	if err != nil {
		t.Fatalf("save vectors: %v", err)
	}

	eng := &Engine{
		Store:        st,
		TextEmbedder: &fakeTextEmbedder{},
		VisualEmbed:  &fakeVisualEmbedder{},
	}
	res, err := eng.Search(context.Background(), "v1", "error trace", 2)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if res.Status != "ok" || len(res.Segments) != 2 {
		t.Fatalf("unexpected search result: %+v", res)
	}
	for _, seg := range res.Segments {
		if seg.Verified {
			t.Fatalf("RAG hits must be unverified")
		}
		if len(seg.Matches) != 2 {
			t.Fatalf("expected 2 channel matches for %s, got %d", seg.SegmentID, len(seg.Matches))
		}
	}
}
