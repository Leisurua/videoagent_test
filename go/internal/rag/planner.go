package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/store"
)

// NewLLMPlanner creates an IndexPlanner backed by a ChatReasoner.
// It inspects segment speech_coverage, visual_change, and ASR text to decide
// whether a secondary multimodal visual index is warranted.
func NewLLMPlanner(reasoner cloud.ChatReasoner) IndexPlanner {
	return func(ctx context.Context, segments []store.SegmentRecord) (IndexDecision, error) {
		if reasoner == nil {
			return IndexDecision{BuildVisualIndex: false, Reason: "planner reasoner not configured"}, nil
		}
		type segSummary struct {
			SegmentID      string  `json:"segment_id"`
			StartMs        int64   `json:"start_ms"`
			EndMs          int64   `json:"end_ms"`
			SpeechCoverage float64 `json:"speech_coverage"`
			VisualChange   float64 `json:"visual_change"`
			ASRPreview     string  `json:"asr_preview"`
		}
		summaries := make([]segSummary, 0, len(segments))
		for _, s := range segments {
			asr := strings.TrimSpace(s.ASR)
			if len(asr) > 120 {
				asr = asr[:120] + "..."
			}
			summaries = append(summaries, segSummary{
				SegmentID:      s.SegmentID,
				StartMs:        s.StartMs,
				EndMs:          s.EndMs,
				SpeechCoverage: s.SpeechCoverage,
				VisualChange:   s.VisualChange,
				ASRPreview:     asr,
			})
		}
		rawSegs, _ := json.Marshal(summaries)
		sys := `You are a Video Index Planner. Decide whether to build a secondary multimodal visual vector index based on segment speech_coverage, visual_change, and ASR previews.
Return ONLY JSON: {"build_visual_index": bool, "reason": "non-empty reason", "candidate_segments": ["seg_0000", ...]}. An empty candidate_segments list when build_visual_index=true means index the whole video.`
		usr := fmt.Sprintf("Segments:\n%s", string(rawSegs))
		resp, err := reasoner.Chat(ctx, cloud.ChatRequest{
			Messages: []cloud.ChatMessage{
				{Role: "system", Content: &sys},
				{Role: "user", Content: &usr},
			},
		})
		if err != nil {
			return IndexDecision{}, err
		}
		if resp.Message.Content == nil {
			return IndexDecision{}, fmt.Errorf("planner returned empty content")
		}
		txt := strings.TrimSpace(*resp.Message.Content)
		if start := strings.Index(txt, "{"); start >= 0 {
			if end := strings.LastIndex(txt, "}"); end > start {
				txt = txt[start : end+1]
			}
		}
		var dec IndexDecision
		if err := json.Unmarshal([]byte(txt), &dec); err != nil {
			return IndexDecision{}, fmt.Errorf("parse planner decision: %w", err)
		}
		if strings.TrimSpace(dec.Reason) == "" {
			dec.Reason = "planner decision"
		}
		return dec, nil
	}
}
