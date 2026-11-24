package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/media"
	mediapb "github.com/videoagent/videoagent_project/go/internal/media/pb"
	"github.com/videoagent/videoagent_project/go/internal/rag"
	"github.com/videoagent/videoagent_project/go/internal/store"
)

// VideoTools binds a single indexed video to the 9 atomic ReAct tools.
// Tools contain no budget or guard logic; ExecutionGuard enforces all limits.
type VideoTools struct {
	VideoID     string
	VideoPath   string
	DurationMs  int64
	ArtifactDir string
	Store       *store.Store
	RAG         *rag.Engine
	Media       *media.Client
	Vision      cloud.VisionObserver
}

func (t *VideoTools) Definitions() []cloud.ToolDefinition {
	return []cloud.ToolDefinition{
		def("search_segments", "Search indexed ASR and visual segments using dual-channel RAG with RRF fusion. Returns candidate segments (unverified).", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "Search query"},
				"top_k": map[string]any{"type": "integer", "description": "Max segments to return (default 5)"},
			},
			"required": []string{"query"},
		}),
		def("get_segment", "Read full ASR text, visual description, and metadata for one segment_id.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"segment_id": map[string]any{"type": "string"},
			},
			"required": []string{"segment_id"},
		}),
		def("get_neighbors", "Read neighboring segments before and after segment_id in chronological order.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"segment_id": map[string]any{"type": "string"},
				"before":     map[string]any{"type": "integer"},
				"after":      map[string]any{"type": "integer"},
			},
			"required": []string{"segment_id"},
		}),
		def("inspect_segment", "Sample up to 8 frames from the original video in [start_ms, end_ms) and ask the VLM an objective visual question.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"start_ms": map[string]any{"type": "integer"},
				"end_ms":   map[string]any{"type": "integer"},
				"question": map[string]any{"type": "string"},
			},
			"required": []string{"start_ms", "end_ms", "question"},
		}),
		def("locate_event", "Coarse-to-fine search for a visual event in [start_ms, end_ms) (max 300000ms). Returns bracketed interval and frame refs.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"start_ms":     map[string]any{"type": "integer"},
				"end_ms":       map[string]any{"type": "integer"},
				"query":        map[string]any{"type": "string"},
				"precision_ms": map[string]any{"type": "integer"},
			},
			"required": []string{"start_ms", "end_ms", "query"},
		}),
		def("get_frame", "Extract the original-resolution frame displayed at timestamp_ms as a PNG artifact.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"timestamp_ms": map[string]any{"type": "integer"},
			},
			"required": []string{"timestamp_ms"},
		}),
		def("inspect_frame", "Extract the frame at timestamp_ms and ask the VLM an objective visual question.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"timestamp_ms": map[string]any{"type": "integer"},
				"question":     map[string]any{"type": "string"},
			},
			"required": []string{"timestamp_ms", "question"},
		}),
		def("ocr_frame", "Run Tesseract OCR on the frame at timestamp_ms (optionally cropped to normalized bbox [x1,y1,x2,y2]).", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"timestamp_ms": map[string]any{"type": "integer"},
				"bbox":         map[string]any{"type": "array", "items": map[string]any{"type": "number"}},
				"language":     map[string]any{"type": "string"},
			},
			"required": []string{"timestamp_ms"},
		}),
		def("crop_frame", "Crop the frame at timestamp_ms to normalized bbox [x1,y1,x2,y2] and return original + cropped PNG artifacts.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"timestamp_ms": map[string]any{"type": "integer"},
				"bbox":         map[string]any{"type": "array", "items": map[string]any{"type": "number"}},
			},
			"required": []string{"timestamp_ms", "bbox"},
		}),
	}
}

func def(name, description string, params any) cloud.ToolDefinition {
	var td cloud.ToolDefinition
	td.Type = "function"
	td.Function.Name = name
	td.Function.Description = description
	td.Function.Parameters = params
	return td
}

// Execute runs one atomic tool by name and returns a JSON-serializable map.
func (t *VideoTools) Execute(ctx context.Context, name, argsJSON string) (map[string]any, error) {
	switch name {
	case "search_segments":
		var a struct {
			Query string `json:"query"`
			TopK  int    `json:"top_k"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		res, err := t.RAG.Search(ctx, t.VideoID, a.Query, a.TopK)
		if err != nil {
			return nil, err
		}
		return toMap(res)

	case "get_segment":
		var a struct {
			SegmentID string `json:"segment_id"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		seg, err := t.Store.GetSegment(t.VideoID, a.SegmentID)
		if err != nil {
			return nil, err
		}
		if seg == nil {
			return nil, fmt.Errorf("segment %q not found", a.SegmentID)
		}
		return map[string]any{
			"segment":  seg,
			"source":   "indexed_evidence",
			"verified": false,
		}, nil

	case "get_neighbors":
		var a struct {
			SegmentID string `json:"segment_id"`
			Before    int    `json:"before"`
			After     int    `json:"after"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		segs, err := t.Store.ListSegments(t.VideoID)
		if err != nil {
			return nil, err
		}
		idx := -1
		for i, s := range segs {
			if s.SegmentID == a.SegmentID {
				idx = i
				break
			}
		}
		if idx < 0 {
			return nil, fmt.Errorf("segment %q not found", a.SegmentID)
		}
		if a.Before < 0 {
			a.Before = 1
		}
		if a.After < 0 {
			a.After = 1
		}
		start := idx - a.Before
		if start < 0 {
			start = 0
		}
		end := idx + a.After + 1
		if end > len(segs) {
			end = len(segs)
		}
		var neighbors []store.SegmentRecord
		for i := start; i < end; i++ {
			if i != idx {
				neighbors = append(neighbors, segs[i])
			}
		}
		return map[string]any{
			"segments": neighbors,
			"source":   "indexed_evidence",
			"verified": false,
		}, nil

	case "get_frame":
		var a struct {
			TimestampMs int64 `json:"timestamp_ms"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		resp, err := t.Media.GetFrame(ctx, &mediapb.GetFrameRequest{
			VideoId:     t.VideoID,
			VideoPath:   t.VideoPath,
			TimestampMs: a.TimestampMs,
			ArtifactDir: t.ArtifactDir,
		})
		if err != nil {
			return nil, err
		}
		return toMap(resp)

	case "inspect_frame":
		var a struct {
			TimestampMs int64  `json:"timestamp_ms"`
			Question    string `json:"question"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		fResp, err := t.Media.GetFrame(ctx, &mediapb.GetFrameRequest{
			VideoId:     t.VideoID,
			VideoPath:   t.VideoPath,
			TimestampMs: a.TimestampMs,
			ArtifactDir: t.ArtifactDir,
		})
		if err != nil {
			return nil, err
		}
		f := fResp.GetFrame()
		if t.Vision == nil {
			return map[string]any{
				"status": "not_configured",
				"answer": nil,
				"frames": []any{f},
			}, nil
		}
		vres, err := t.Vision.Observe(ctx, cloud.VisionRequest{
			Task:     "inspect",
			Question: a.Question,
			StartMs:  f.GetPtsMs(),
			EndMs:    f.GetDisplayEndMs(),
			Frames: []cloud.FrameInput{{
				Ref:       f.GetRef(),
				ImagePath: f.GetImagePath(),
				PTSMs:     f.GetPtsMs(),
			}},
		})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error(), "frames": []any{f}}, nil
		}
		return map[string]any{
			"status":        vres.Status,
			"answer":        vres.Answer,
			"evidence_refs": vres.EvidenceRefs,
			"frames":        []any{f},
			"start_ms":      f.GetPtsMs(),
			"end_ms":        f.GetDisplayEndMs(),
			"source":        "inspect_frame",
		}, nil

	case "inspect_segment":
		var a struct {
			StartMs  int64  `json:"start_ms"`
			EndMs    int64  `json:"end_ms"`
			Question string `json:"question"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		sResp, err := t.Media.SampleFrames(ctx, &mediapb.SampleFramesRequest{
			VideoId:     t.VideoID,
			VideoPath:   t.VideoPath,
			StartMs:     a.StartMs,
			EndMs:       a.EndMs,
			NumFrames:   8,
			ArtifactDir: t.ArtifactDir,
		})
		if err != nil {
			return nil, err
		}
		overlap, _ := t.overlappingSegments(a.StartMs, a.EndMs)
		if t.Vision == nil {
			return map[string]any{
				"status":      "not_configured",
				"answer":      nil,
				"segment_ids": overlap,
				"frames":      sResp.GetFrames(),
			}, nil
		}
		inputs := make([]cloud.FrameInput, 0, len(sResp.GetFrames()))
		for _, f := range sResp.GetFrames() {
			inputs = append(inputs, cloud.FrameInput{
				Ref:       f.GetRef(),
				ImagePath: f.GetImagePath(),
				PTSMs:     f.GetPtsMs(),
			})
		}
		vres, err := t.Vision.Observe(ctx, cloud.VisionRequest{
			Task:     "inspect",
			Question: a.Question,
			StartMs:  a.StartMs,
			EndMs:    a.EndMs,
			Frames:   inputs,
		})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error(), "frames": sResp.GetFrames()}, nil
		}
		return map[string]any{
			"status":        vres.Status,
			"answer":        vres.Answer,
			"evidence_refs": vres.EvidenceRefs,
			"segment_ids":   overlap,
			"frames":        sResp.GetFrames(),
			"start_ms":      a.StartMs,
			"end_ms":        a.EndMs,
			"source":        "inspect_segment",
		}, nil

	case "locate_event":
		var a struct {
			StartMs     int64  `json:"start_ms"`
			EndMs       int64  `json:"end_ms"`
			Query       string `json:"query"`
			PrecisionMs int64  `json:"precision_ms"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		return t.locateEvent(ctx, a.StartMs, a.EndMs, a.Query, a.PrecisionMs)

	case "crop_frame":
		var a struct {
			TimestampMs int64     `json:"timestamp_ms"`
			Bbox        []float64 `json:"bbox"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil || len(a.Bbox) != 4 {
			return nil, fmt.Errorf("crop_frame requires timestamp_ms and bbox [x1,y1,x2,y2]")
		}
		resp, err := t.Media.CropFrame(ctx, &mediapb.CropFrameRequest{
			VideoId:     t.VideoID,
			VideoPath:   t.VideoPath,
			TimestampMs: a.TimestampMs,
			Bbox: &mediapb.NormalizedBBox{
				X1: a.Bbox[0], Y1: a.Bbox[1], X2: a.Bbox[2], Y2: a.Bbox[3],
			},
			ArtifactDir: t.ArtifactDir,
		})
		if err != nil {
			return nil, err
		}
		return toMap(resp)

	case "ocr_frame":
		var a struct {
			TimestampMs int64     `json:"timestamp_ms"`
			Bbox        []float64 `json:"bbox"`
			Language    string    `json:"language"`
		}
		if err := json.Unmarshal([]byte(argsJSON), &a); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
		req := &mediapb.OCRFrameRequest{
			VideoId:     t.VideoID,
			VideoPath:   t.VideoPath,
			TimestampMs: a.TimestampMs,
			Language:    a.Language,
			ArtifactDir: t.ArtifactDir,
		}
		if len(a.Bbox) == 4 {
			req.Bbox = &mediapb.NormalizedBBox{
				X1: a.Bbox[0], Y1: a.Bbox[1], X2: a.Bbox[2], Y2: a.Bbox[3],
			}
		}
		resp, err := t.Media.OCRFrame(ctx, req)
		if err != nil {
			return nil, err
		}
		return toMap(resp)

	default:
		return nil, fmt.Errorf("unknown tool %q", name)
	}
}

func (t *VideoTools) locateEvent(ctx context.Context, startMs, endMs int64, query string, precisionMs int64) (map[string]any, error) {
	if endMs <= startMs || startMs < 0 {
		return nil, fmt.Errorf("invalid range [%d, %d)", startMs, endMs)
	}
	if endMs-startMs > 300_000 {
		return map[string]any{"status": "limit_exceeded", "max_span_ms": 300_000}, nil
	}
	if t.Vision == nil {
		return map[string]any{"status": "not_configured"}, nil
	}
	if precisionMs <= 0 {
		precisionMs = 1000
	}

	curS, curE := startMs, endMs
	var bestFrames []*mediapb.FrameArtifact
	var bestRefs []string
	var lastAnswer string
	foundOnce := false

	for round := 0; round < 4; round++ {
		sResp, err := t.Media.SampleFrames(ctx, &mediapb.SampleFramesRequest{
			VideoId:     t.VideoID,
			VideoPath:   t.VideoPath,
			StartMs:     curS,
			EndMs:       curE,
			NumFrames:   8,
			ArtifactDir: t.ArtifactDir,
		})
		if err != nil {
			return nil, err
		}
		frames := sResp.GetFrames()
		inputs := make([]cloud.FrameInput, 0, len(frames))
		frameByRef := make(map[string]*mediapb.FrameArtifact, len(frames))
		for _, f := range frames {
			inputs = append(inputs, cloud.FrameInput{Ref: f.GetRef(), ImagePath: f.GetImagePath(), PTSMs: f.GetPtsMs()})
			frameByRef[f.GetRef()] = f
		}
		vres, err := t.Vision.Observe(ctx, cloud.VisionRequest{
			Task:     "locate_event",
			Question: query,
			StartMs:  curS,
			EndMs:    curE,
			Frames:   inputs,
		})
		if err != nil {
			return map[string]any{"status": "error", "error": err.Error()}, nil
		}
		if vres.Status != "matched" || len(vres.EvidenceRefs) == 0 {
			if foundOnce {
				return map[string]any{
					"status":        "uncertain",
					"start_ms":      curS,
					"end_ms":        curE,
					"evidence_refs": bestRefs,
					"frames":        bestFrames,
					"time_basis":    "inferred_from_sampled_frames",
					"answer":        lastAnswer,
				}, nil
			}
			return map[string]any{
				"status":   "not_found",
				"start_ms": startMs,
				"end_ms":   endMs,
			}, nil
		}

		foundOnce = true
		bestRefs = vres.EvidenceRefs
		bestFrames = frames
		lastAnswer = vres.Answer

		// Bracket around matched frames.
		var matchedPts []int64
		for _, ref := range vres.EvidenceRefs {
			if f, ok := frameByRef[ref]; ok {
				matchedPts = append(matchedPts, f.GetPtsMs())
			}
		}
		if len(matchedPts) == 0 {
			break
		}
		sort.Slice(matchedPts, func(i, j int) bool { return matchedPts[i] < matchedPts[j] })
		firstHit := matchedPts[0]
		newS := curS
		for _, f := range frames {
			if f.GetPtsMs() < firstHit {
				newS = f.GetPtsMs()
			}
		}
		newE := matchedPts[len(matchedPts)-1]
		if newE <= newS {
			newE = frames[len(frames)-1].GetDisplayEndMs()
		}
		curS, curE = newS, newE
		if curE-curS <= precisionMs {
			break
		}
	}

	achieved := curE - curS
	return map[string]any{
		"status":                "found",
		"start_ms":              curS,
		"end_ms":                curE,
		"achieved_precision_ms": achieved,
		"precision_reached":     achieved <= precisionMs,
		"evidence_refs":         bestRefs,
		"frames":                bestFrames,
		"answer":                lastAnswer,
		"time_basis":            "inferred_from_sampled_frames",
		"source":                "locate_event",
	}, nil
}

func (t *VideoTools) overlappingSegments(startMs, endMs int64) ([]string, error) {
	segs, err := t.Store.ListSegments(t.VideoID)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, s := range segs {
		if s.EndMs > startMs && s.StartMs < endMs {
			ids = append(ids, s.SegmentID)
		}
	}
	return ids, nil
}

func toMap(v any) (map[string]any, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return m, nil
}
