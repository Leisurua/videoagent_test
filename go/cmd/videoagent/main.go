package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/videoagent/videoagent_project/go/internal/agent"
	"github.com/videoagent/videoagent_project/go/internal/cloud"
	"github.com/videoagent/videoagent_project/go/internal/eval"
	"github.com/videoagent/videoagent_project/go/internal/media"
	mediapb "github.com/videoagent/videoagent_project/go/internal/media/pb"
	"github.com/videoagent/videoagent_project/go/internal/rag"
	"github.com/videoagent/videoagent_project/go/internal/router"
	"github.com/videoagent/videoagent_project/go/internal/store"
	"github.com/videoagent/videoagent_project/go/internal/tools"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:50051", "gRPC address of C++ media_daemon")
	artifactDir := flag.String("artifact-dir", "data/evidence", "directory for PNG artifacts")
	dbPath := flag.String("db", "data/videoagent.db", "SQLite database path")
	asrModel := flag.String("asr-model", "", "path to whisper.cpp ggml model (.bin)")
	embedBase := flag.String("embed-base", "https://api.siliconflow.cn/v1", "OpenAI-compatible embedding base URL")
	embedModel := flag.String("embed-model", "BAAI/bge-m3", "cloud text embedding model")
	visEmbedModel := flag.String("vis-embed-model", "", "optional cloud multimodal visual embedding model")
	maxWorkers := flag.Int("workers", 2, "max concurrent Workers for Super-Agent Router")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		printUsage()
		os.Exit(2)
	}

	ctx := context.Background()
	client, err := media.Dial(ctx, *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	switch args[0] {
	case "health":
		resp, err := client.HealthCheck(ctx)
		must(err)
		printJSON(resp)

	case "probe":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: videoagent probe <video_path>")
			os.Exit(2)
		}
		resp, err := client.GetVideoMetadata(ctx, args[1])
		must(err)
		printJSON(resp)

	case "frame":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: videoagent frame <video_path> <timestamp_ms> [video_id]")
			os.Exit(2)
		}
		ts := mustInt64(args[2], "timestamp_ms")
		vid := optArg(args, 3, "demo")
		resp, err := client.GetFrame(ctx, &mediapb.GetFrameRequest{
			VideoId:     vid,
			VideoPath:   args[1],
			TimestampMs: ts,
			ArtifactDir: *artifactDir,
		})
		must(err)
		printJSON(resp)

	case "sample":
		if len(args) < 5 {
			fmt.Fprintln(os.Stderr, "usage: videoagent sample <video_path> <start_ms> <end_ms> <num_frames> [w h]")
			os.Exit(2)
		}
		startMs := mustInt64(args[2], "start_ms")
		endMs := mustInt64(args[3], "end_ms")
		n := int32(mustInt64(args[4], "num_frames"))
		var w, h int32
		if len(args) >= 7 {
			w = int32(mustInt64(args[5], "width"))
			h = int32(mustInt64(args[6], "height"))
		}
		resp, err := client.SampleFrames(ctx, &mediapb.SampleFramesRequest{
			VideoId:      "demo",
			VideoPath:    args[1],
			StartMs:      startMs,
			EndMs:        endMs,
			NumFrames:    n,
			TargetWidth:  w,
			TargetHeight: h,
			ArtifactDir:  *artifactDir,
		})
		must(err)
		printJSON(resp)

	case "crop":
		if len(args) < 7 {
			fmt.Fprintln(os.Stderr, "usage: videoagent crop <video_path> <timestamp_ms> <x1> <y1> <x2> <y2>")
			os.Exit(2)
		}
		ts := mustInt64(args[2], "timestamp_ms")
		resp, err := client.CropFrame(ctx, &mediapb.CropFrameRequest{
			VideoId:     "demo",
			VideoPath:   args[1],
			TimestampMs: ts,
			Bbox: &mediapb.NormalizedBBox{
				X1: mustFloat64(args[3], "x1"),
				Y1: mustFloat64(args[4], "y1"),
				X2: mustFloat64(args[5], "x2"),
				Y2: mustFloat64(args[6], "y2"),
			},
			ArtifactDir: *artifactDir,
		})
		must(err)
		printJSON(resp)

	case "asr":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: videoagent asr <video_path> <ggml_model_path> [language]")
			os.Exit(2)
		}
		lang := optArg(args, 3, "auto")
		resp, err := client.TranscribeASR(ctx, &mediapb.TranscribeASRRequest{
			VideoPath: args[1],
			ModelPath: args[2],
			Language:  lang,
		})
		must(err)
		printJSON(resp)

	case "ocr":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: videoagent ocr <video_path> <timestamp_ms> [language]")
			os.Exit(2)
		}
		ts := mustInt64(args[2], "timestamp_ms")
		lang := optArg(args, 3, "eng")
		resp, err := client.OCRFrame(ctx, &mediapb.OCRFrameRequest{
			VideoId:     "demo",
			VideoPath:   args[1],
			TimestampMs: ts,
			Language:    lang,
			ArtifactDir: *artifactDir,
		})
		must(err)
		printJSON(resp)

	case "index":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: videoagent index <video_path> [video_id]")
			os.Exit(2)
		}
		st, err := store.Open(*dbPath)
		must(err)
		defer st.Close()

		vid := optArg(args, 2, "")
		if vid == "" {
			vid, err = videoContentID(args[1])
			must(err)
		}

		var embedder cloud.TextEmbedder
		if key := firstEnv("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"); key != "" {
			embedder = cloud.NewOpenAITextEmbedder(*embedBase, key, *embedModel, 60*time.Second)
		}
		var visEmbedder cloud.VisualEmbedder
		if *visEmbedModel != "" {
			if key := firstEnv("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"); key != "" {
				visEmbedder = cloud.NewMultimodalEmbedder(*embedBase, key, *visEmbedModel, 90*time.Second)
			}
		}
		var planner rag.IndexPlanner
		if dsKey := os.Getenv("DEEPSEEK_API_KEY"); dsKey != "" && visEmbedder != nil {
			planner = rag.NewLLMPlanner(cloud.NewDeepSeekClient(dsKey, "deepseek-chat", 60*time.Second))
		}
		eng := &rag.Engine{Store: st, Media: client, TextEmbedder: embedder, VisualEmbed: visEmbedder}
		rec, err := eng.IndexVideo(ctx, rag.IndexConfig{
			VideoID:      vid,
			VideoPath:    args[1],
			ASRModelPath: *asrModel,
			ASRLanguage:  "auto",
			ArtifactDir:  *artifactDir,
			TextEmbedder: embedder,
			VisualEmbed:  visEmbedder,
			Planner:      planner,
		})
		must(err)
		printJSON(rec)

	case "ask":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: videoagent ask <video_id> <question>")
			os.Exit(2)
		}
		deepseekKey := os.Getenv("DEEPSEEK_API_KEY")
		if deepseekKey == "" {
			fmt.Fprintln(os.Stderr, "error: DEEPSEEK_API_KEY is required")
			os.Exit(1)
		}
		st, err := store.Open(*dbPath)
		must(err)
		defer st.Close()

		vidRec, err := st.GetVideo(args[1])
		must(err)
		if vidRec == nil {
			fmt.Fprintf(os.Stderr, "error: video_id %q not indexed in %s\n", args[1], *dbPath)
			os.Exit(1)
		}

		var embedder cloud.TextEmbedder
		if key := firstEnv("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"); key != "" {
			embedder = cloud.NewOpenAITextEmbedder(*embedBase, key, *embedModel, 60*time.Second)
		}
		var vision cloud.VisionObserver
		if sfKey := os.Getenv("SILICONFLOW_API_KEY"); sfKey != "" {
			vision = cloud.NewSiliconFlowVision(sfKey, "Qwen/Qwen3.8-27B", 90*time.Second)
		}

		eng := &rag.Engine{Store: st, Media: client, TextEmbedder: embedder}
		vt := &tools.VideoTools{
			VideoID:     vidRec.VideoID,
			VideoPath:   vidRec.VideoPath,
			DurationMs:  vidRec.DurationMs,
			ArtifactDir: *artifactDir,
			Store:       st,
			RAG:         eng,
			Media:       client,
			Vision:      vision,
		}
		r := &router.Router{
			Reasoner:     cloud.NewDeepSeekClient(deepseekKey, "deepseek-chat", 120*time.Second),
			Store:        st,
			Tools:        vt,
			WorkerConfig: agent.DefaultConfig(),
			MaxWorkers:   *maxWorkers,
		}
		res, err := r.Ask(ctx, vidRec.VideoID, args[2])
		must(err)
		printJSON(res)

	case "eval":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: videoagent eval <video_id> <questions.json> [A,B,C,D]")
			os.Exit(2)
		}
		deepseekKey := os.Getenv("DEEPSEEK_API_KEY")
		if deepseekKey == "" {
			fmt.Fprintln(os.Stderr, "error: DEEPSEEK_API_KEY is required")
			os.Exit(1)
		}
		rawQ, err := os.ReadFile(args[2])
		must(err)
		var questions []eval.QuestionItem
		must(json.Unmarshal(rawQ, &questions))

		var groups []eval.Group
		if len(args) >= 4 && args[3] != "" {
			for _, g := range strings.Split(args[3], ",") {
				groups = append(groups, eval.Group(strings.TrimSpace(g)))
			}
		}

		st, err := store.Open(*dbPath)
		must(err)
		defer st.Close()
		vidRec, err := st.GetVideo(args[1])
		must(err)
		if vidRec == nil {
			fmt.Fprintf(os.Stderr, "error: video_id %q not indexed in %s\n", args[1], *dbPath)
			os.Exit(1)
		}

		var embedder cloud.TextEmbedder
		if key := firstEnv("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"); key != "" {
			embedder = cloud.NewOpenAITextEmbedder(*embedBase, key, *embedModel, 60*time.Second)
		}
		var visEmbedder cloud.VisualEmbedder
		if *visEmbedModel != "" {
			if key := firstEnv("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"); key != "" {
				visEmbedder = cloud.NewMultimodalEmbedder(*embedBase, key, *visEmbedModel, 90*time.Second)
			}
		}
		var vision cloud.VisionObserver
		if sfKey := os.Getenv("SILICONFLOW_API_KEY"); sfKey != "" {
			vision = cloud.NewSiliconFlowVision(sfKey, "Qwen/Qwen3.8-27B", 90*time.Second)
		}

		textRAG := &rag.Engine{Store: st, Media: client, TextEmbedder: embedder}
		dualRAG := &rag.Engine{Store: st, Media: client, TextEmbedder: embedder, VisualEmbed: visEmbedder}
		vt := &tools.VideoTools{
			VideoID:     vidRec.VideoID,
			VideoPath:   vidRec.VideoPath,
			DurationMs:  vidRec.DurationMs,
			ArtifactDir: *artifactDir,
			Store:       st,
			RAG:         textRAG,
			Media:       client,
			Vision:      vision,
		}
		evRunner := &eval.Runner{
			Reasoner:     cloud.NewDeepSeekClient(deepseekKey, "deepseek-chat", 120*time.Second),
			Store:        st,
			Tools:        vt,
			TextOnlyRAG:  textRAG,
			DualRAG:      dualRAG,
			WorkerConfig: agent.DefaultConfig(),
			MaxWorkers:   *maxWorkers,
		}
		rep, err := evRunner.Evaluate(ctx, vidRec.VideoID, questions, groups)
		must(err)
		printJSON(rep)

	default:
		printUsage()
		os.Exit(2)
	}
}

func printUsage() {
	fmt.Fprintln(os.Stderr, "usage: videoagent [--addr host:port] [--db path] [--artifact-dir dir] <health|probe|frame|sample|crop|asr|ocr|index|ask> ...")
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

func videoContentID(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "vid-" + hex.EncodeToString(h.Sum(nil))[:24], nil
}

func must(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func optArg(args []string, idx int, def string) string {
	if len(args) > idx {
		return args[idx]
	}
	return def
}

func mustInt64(s, name string) int64 {
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid %s %q: %v\n", name, s, err)
		os.Exit(2)
	}
	return v
}

func mustFloat64(s, name string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid %s %q: %v\n", name, s, err)
		os.Exit(2)
	}
	return v
}

func printJSON(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "encode json: %v\n", err)
		os.Exit(1)
	}
}
