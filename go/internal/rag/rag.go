package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/media"
	mediapb "github.com/videoagent/videoagent_project/go/internal/media/pb"
	"github.com/videoagent/videoagent_project/go/internal/store"
)

const DefaultSegmentMs int64 = 30_000

// ASRSpan is a timestamped speech segment in integer milliseconds.
type ASRSpan struct {
	StartMs int64  `json:"start_ms"`
	EndMs   int64  `json:"end_ms"`
	Text    string `json:"text"`
}

// IndexDecision is returned by an optional IndexPlanner after text indexing.
type IndexDecision struct {
	BuildVisualIndex  bool     `json:"build_visual_index"`
	Reason            string   `json:"reason"`
	CandidateSegments []string `json:"candidate_segments"`
}

// IndexPlanner decides whether to build the secondary multimodal visual index.
type IndexPlanner func(ctx context.Context, segments []store.SegmentRecord) (IndexDecision, error)

// IndexConfig controls video initialization and dual-channel indexing.
type IndexConfig struct {
	VideoID      string
	VideoPath    string
	SegmentMs    int64
	ASRModelPath string
	ASRLanguage  string
	PrecomputedASR []ASRSpan
	ArtifactDir  string
	TextEmbedder cloud.TextEmbedder
	VisualEmbed  cloud.VisualEmbedder
	Planner      IndexPlanner
}

// SearchMatch records a segment's hit in one retrieval channel.
type SearchMatch struct {
	Channel  string  `json:"channel"`
	Rank     int     `json:"rank"`
	Distance float64 `json:"distance"`
}

// SearchHit is a deduplicated RRF-fused candidate segment.
type SearchHit struct {
	SegmentID string              `json:"segment_id"`
	StartMs   int64               `json:"start_ms"`
	EndMs     int64               `json:"end_ms"`
	ASR       string              `json:"asr"`
	Visual    string              `json:"visual"`
	Score     float64             `json:"score"`
	Source    string              `json:"source"`
	Verified  bool                `json:"verified"`
	Matches   []SearchMatch       `json:"matches"`
}

// SearchResult is returned by DualChannelRAG.Search.
type SearchResult struct {
	Status             string      `json:"status"` // "ok" or "partial"
	UnavailableSources []string    `json:"unavailable_sources,omitempty"`
	Segments           []SearchHit `json:"segments"`
}

// Engine manages dual-channel indexing and RRF search over SQLite + gRPC media_daemon.
type Engine struct {
	Store        *store.Store
	Media        *media.Client
	TextEmbedder cloud.TextEmbedder
	VisualEmbed  cloud.VisualEmbedder
}

// IndexVideo builds the 30s segment index, text embeddings, and optional visual embeddings.
func (e *Engine) IndexVideo(ctx context.Context, cfg IndexConfig) (*store.VideoRecord, error) {
	if existing, err := e.Store.GetVideo(cfg.VideoID); err != nil {
		return nil, err
	} else if existing != nil {
		return nil, fmt.Errorf("video_id %q already indexed; use a new video_id to rebuild", cfg.VideoID)
	}

	meta, err := e.Media.GetVideoMetadata(ctx, cfg.VideoPath)
	if err != nil {
		return nil, fmt.Errorf("get video metadata: %w", err)
	}

	segMs := cfg.SegmentMs
	if segMs <= 0 {
		segMs = DefaultSegmentMs
	}

	asrSpans := cfg.PrecomputedASR
	if len(asrSpans) == 0 && cfg.ASRModelPath != "" {
		resp, err := e.Media.TranscribeASR(ctx, &mediapb.TranscribeASRRequest{
			VideoPath: cfg.VideoPath,
			ModelPath: cfg.ASRModelPath,
			Language:  cfg.ASRLanguage,
		})
		if err != nil {
			return nil, fmt.Errorf("transcribe ASR: %w", err)
		}
		for _, sp := range resp.GetSpans() {
			asrSpans = append(asrSpans, ASRSpan{
				StartMs: sp.GetStartMs(),
				EndMs:   sp.GetEndMs(),
				Text:    sp.GetText(),
			})
		}
	}

	var segments []store.SegmentRecord
	for idx, start := 0, int64(0); start < meta.GetDurationMs(); idx, start = idx+1, start+segMs {
		end := start + segMs
		if end > meta.GetDurationMs() {
			end = meta.GetDurationMs()
		}
		segID := fmt.Sprintf("seg_%04d", idx)
		asrText, coverage := collectWindowASR(asrSpans, start, end)

		sampleResp, err := e.Media.SampleFrames(ctx, &mediapb.SampleFramesRequest{
			VideoId:      cfg.VideoID,
			VideoPath:    cfg.VideoPath,
			StartMs:      start,
			EndMs:        end,
			NumFrames:    3,
			TargetWidth:  320,
			TargetHeight: 180,
			ArtifactDir:  cfg.ArtifactDir,
		})
		if err != nil {
			return nil, fmt.Errorf("sample frames for %s: %w", segID, err)
		}
		ptsList := make([]int64, 0, len(sampleResp.GetFrames()))
		for _, f := range sampleResp.GetFrames() {
			ptsList = append(ptsList, f.GetPtsMs())
		}

		segments = append(segments, store.SegmentRecord{
			VideoID:             cfg.VideoID,
			SegmentID:           segID,
			StartMs:             start,
			EndMs:               end,
			ASR:                 asrText,
			Visual:              "",
			SpeechCoverage:      coverage,
			VisualChange:        sampleResp.GetVisualChange(),
			VisualStatus:        "not_configured",
			SampledTimestampsMs: ptsList,
		})
	}

	if err := e.Store.SaveSegments(segments); err != nil {
		return nil, err
	}

	// Build text channel embeddings.
	textEmbedder := cfg.TextEmbedder
	if textEmbedder == nil {
		textEmbedder = e.TextEmbedder
	}
	if textEmbedder != nil {
		var idxToEmbed []int
		var texts []string
		for i, seg := range segments {
			combined := formatRetrievalText(seg.ASR, seg.Visual)
			if combined != "" {
				idxToEmbed = append(idxToEmbed, i)
				texts = append(texts, combined)
			}
		}
		if len(texts) > 0 {
			vecs, err := textEmbedder.Embed(ctx, texts)
			if err != nil {
				return nil, fmt.Errorf("embed text segments: %w", err)
			}
			var records []store.VectorRecord
			for j, segIdx := range idxToEmbed {
				records = append(records, store.VectorRecord{
					VideoID:   cfg.VideoID,
					Channel:   "text",
					SegmentID: segments[segIdx].SegmentID,
					ModelID:   textEmbedder.ModelID(),
					Vector:    vecs[j],
				})
			}
			if err := e.Store.SaveVectors(records); err != nil {
				return nil, err
			}
		}
	}

	// Optional Planner & secondary visual channel.
	status := "ready_text_only"
	plannerStatus := "not_configured"
	plannerReason := ""
	buildVisual := false

	if cfg.Planner != nil {
		dec, err := cfg.Planner(ctx, segments)
		if err != nil {
			status = "planner_failed"
			plannerStatus = "failed"
			plannerReason = err.Error()
		} else {
			plannerStatus = "decided"
			plannerReason = dec.Reason
			buildVisual = dec.BuildVisualIndex
			if buildVisual {
				visEmbedder := cfg.VisualEmbed
				if visEmbedder == nil {
					visEmbedder = e.VisualEmbed
				}
				if visEmbedder == nil {
					status = "pending_visual_index"
				} else if cfg.ArtifactDir == "" {
					status = "pending_visual_index"
				} else {
					if err := e.buildVisualVectors(ctx, cfg.VideoID, cfg.VideoPath, segments, dec.CandidateSegments, cfg.ArtifactDir, visEmbedder); err != nil {
						status = "visual_index_failed"
					} else {
						status = "ready_multimodal"
					}
				}
			}
		}
	}

	statsBytes, _ := json.Marshal(map[string]any{
		"segment_count": len(segments),
		"duration_ms":   meta.GetDurationMs(),
	})
	rec := store.VideoRecord{
		VideoID:          cfg.VideoID,
		VideoPath:        cfg.VideoPath,
		DurationMs:       meta.GetDurationMs(),
		Width:            meta.GetWidth(),
		Height:           meta.GetHeight(),
		FPS:              meta.GetFps(),
		Status:           status,
		PlannerStatus:    plannerStatus,
		PlannerReason:    plannerReason,
		BuildVisualIndex: buildVisual,
		StatsJSON:        string(statsBytes),
	}
	if err := e.Store.SaveVideo(rec); err != nil {
		return nil, err
	}
	return &rec, nil
}

func (e *Engine) buildVisualVectors(
	ctx context.Context,
	videoID, videoPath string,
	segments []store.SegmentRecord,
	candidates []string,
	artifactDir string,
	vis cloud.VisualEmbedder,
) error {
	selected := make(map[string]bool)
	for _, id := range candidates {
		selected[id] = true
	}
	var targetSegs []store.SegmentRecord
	var imgPaths []string
	for _, seg := range segments {
		if len(selected) > 0 && !selected[seg.SegmentID] {
			continue
		}
		mid := (seg.StartMs + seg.EndMs) / 2
		frameResp, err := e.Media.GetFrame(ctx, &mediapb.GetFrameRequest{
			VideoId:     videoID,
			VideoPath:   videoPath,
			TimestampMs: mid,
			ArtifactDir: artifactDir,
		})
		if err != nil {
			return err
		}
		targetSegs = append(targetSegs, seg)
		imgPaths = append(imgPaths, frameResp.GetFrame().GetImagePath())
	}
	if len(imgPaths) == 0 {
		return nil
	}
	vecs, err := vis.EmbedImages(ctx, imgPaths)
	if err != nil {
		return err
	}
	var records []store.VectorRecord
	for i, seg := range targetSegs {
		records = append(records, store.VectorRecord{
			VideoID:   videoID,
			Channel:   "visual",
			SegmentID: seg.SegmentID,
			ModelID:   vis.ModelID(),
			Vector:    vecs[i],
		})
	}
	return e.Store.SaveVectors(records)
}

// Search performs dual-channel retrieval (text + visual) fused via RRF.
func (e *Engine) Search(ctx context.Context, videoID, query string, topK int) (SearchResult, error) {
	if topK <= 0 {
		topK = 5
	}
	segs, err := e.Store.ListSegments(videoID)
	if err != nil {
		return SearchResult{}, err
	}
	segMap := make(map[string]store.SegmentRecord, len(segs))
	for _, s := range segs {
		segMap[s.SegmentID] = s
	}

	var unavailable []string
	type rrfAcc struct {
		score   float64
		matches []SearchMatch
	}
	fused := make(map[string]*rrfAcc)
	addHits := func(channel string, hits []rankedHit) {
		for rank, h := range hits {
			acc, ok := fused[h.segmentID]
			if !ok {
				acc = &rrfAcc{}
				fused[h.segmentID] = acc
			}
			acc.score += 1.0 / (60.0 + float64(rank+1))
			acc.matches = append(acc.matches, SearchMatch{
				Channel:  channel,
				Rank:     rank + 1,
				Distance: h.distance,
			})
		}
	}

	// Channel A: Text index
	textVecs, err := e.Store.ListVectors(videoID, "text")
	if err != nil {
		return SearchResult{}, err
	}
	if len(textVecs) > 0 {
		if e.TextEmbedder == nil {
			unavailable = append(unavailable, "text")
		} else {
			qVecs, err := e.TextEmbedder.Embed(ctx, []string{query})
			if err != nil {
				return SearchResult{}, fmt.Errorf("embed query: %w", err)
			}
			addHits("text", rankVectors(qVecs[0], textVecs, topK*2))
		}
	}

	// Channel B: Visual index (if built)
	visVecs, err := e.Store.ListVectors(videoID, "visual")
	if err != nil {
		return SearchResult{}, err
	}
	if len(visVecs) > 0 {
		if e.VisualEmbed == nil {
			unavailable = append(unavailable, "visual")
		} else {
			qVecs, err := e.VisualEmbed.EmbedQueries(ctx, []string{query})
			if err != nil {
				return SearchResult{}, fmt.Errorf("embed visual query: %w", err)
			}
			addHits("visual", rankVectors(qVecs[0], visVecs, topK*2))
		}
	}

	hits := make([]SearchHit, 0, len(fused))
	for segID, acc := range fused {
		seg, ok := segMap[segID]
		if !ok {
			continue
		}
		hits = append(hits, SearchHit{
			SegmentID: seg.SegmentID,
			StartMs:   seg.StartMs,
			EndMs:     seg.EndMs,
			ASR:       seg.ASR,
			Visual:    seg.Visual,
			Score:     acc.score,
			Source:    "indexed_evidence",
			Verified:  false,
			Matches:   acc.matches,
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].StartMs < hits[j].StartMs
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}

	status := "ok"
	if len(unavailable) > 0 {
		status = "partial"
	}
	return SearchResult{
		Status:             status,
		UnavailableSources: unavailable,
		Segments:           hits,
	}, nil
}

type rankedHit struct {
	segmentID string
	distance  float64
}

func rankVectors(query []float32, vecs []store.VectorRecord, limit int) []rankedHit {
	hits := make([]rankedHit, 0, len(vecs))
	for _, v := range vecs {
		hits = append(hits, rankedHit{
			segmentID: v.SegmentID,
			distance:  cosineDistance(query, v.Vector),
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		return hits[i].distance < hits[j].distance
	})
	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits
}

func cosineDistance(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	var dot, na, nb float64
	for i := range a {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		na += fa * fa
		nb += fb * fb
	}
	if na == 0 || nb == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(na) * math.Sqrt(nb))
	return 1.0 - sim
}

func formatRetrievalText(asr, visual string) string {
	var parts []string
	if strings.TrimSpace(asr) != "" {
		parts = append(parts, "[ASR] "+strings.TrimSpace(asr))
	}
	if strings.TrimSpace(visual) != "" {
		parts = append(parts, "[Visual] "+strings.TrimSpace(visual))
	}
	return strings.Join(parts, "\n")
}

func collectWindowASR(spans []ASRSpan, startMs, endMs int64) (string, float64) {
	type interval struct{ s, e int64 }
	var texts []string
	var clipped []interval
	for _, sp := range spans {
		if sp.EndMs <= startMs || sp.StartMs >= endMs {
			continue
		}
		if t := strings.TrimSpace(sp.Text); t != "" {
			texts = append(texts, t)
		}
		cs := sp.StartMs
		if cs < startMs {
			cs = startMs
		}
		ce := sp.EndMs
		if ce > endMs {
			ce = endMs
		}
		if ce > cs {
			clipped = append(clipped, interval{cs, ce})
		}
	}
	if len(clipped) == 0 {
		return strings.Join(texts, " "), 0.0
	}
	sort.Slice(clipped, func(i, j int) bool { return clipped[i].s < clipped[j].s })
	var unionMs int64
	curS, curE := clipped[0].s, clipped[0].e
	for _, iv := range clipped[1:] {
		if iv.s <= curE {
			if iv.e > curE {
				curE = iv.e
			}
		} else {
			unionMs += curE - curS
			curS, curE = iv.s, iv.e
		}
	}
	unionMs += curE - curS
	dur := float64(endMs - startMs)
	if dur <= 0 {
		return strings.Join(texts, " "), 0.0
	}
	return strings.Join(texts, " "), float64(unionMs) / dur
}
