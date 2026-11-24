package store

import (
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"time"

	_ "modernc.org/sqlite"
)

// VideoRecord holds video-level metadata and VideoRAG initialization state.
type VideoRecord struct {
	VideoID          string `json:"video_id"`
	VideoPath        string `json:"video_path"`
	DurationMs       int64  `json:"duration_ms"`
	Width            int32  `json:"width"`
	Height           int32  `json:"height"`
	FPS              float64 `json:"fps"`
	Status           string `json:"status"`
	PlannerStatus    string `json:"planner_status"`
	PlannerReason    string `json:"planner_reason"`
	BuildVisualIndex bool   `json:"build_visual_index"`
	StatsJSON        string `json:"stats_json"`
	CreatedAt        int64  `json:"created_at"`
}

// SegmentRecord mirrors the [start_ms, end_ms) segment schema.
type SegmentRecord struct {
	VideoID                 string  `json:"video_id"`
	SegmentID               string  `json:"segment_id"`
	StartMs                 int64   `json:"start_ms"`
	EndMs                   int64   `json:"end_ms"`
	ASR                     string  `json:"asr"`
	Visual                  string  `json:"visual"`
	SpeechCoverage          float64 `json:"speech_coverage"`
	VisualChange            float64 `json:"visual_change"`
	VisualDependency        *int    `json:"visual_dependency,omitempty"`
	DescriptionCompleteness *int    `json:"description_completeness,omitempty"`
	VisualStatus            string  `json:"visual_status"`
	SampledTimestampsMs     []int64 `json:"sampled_timestamps_ms"`
}

// VectorRecord stores a channel-specific embedding vector for a segment.
type VectorRecord struct {
	VideoID   string
	Channel   string // "text" or "visual"
	SegmentID string
	ModelID   string
	Vector    []float32
}

// EvidenceRecord represents a task-oriented verified observation in the Evidence Store.
type EvidenceRecord struct {
	EvidenceID  string   `json:"evidence_id"`
	RunID       string   `json:"run_id"`
	WorkerID    string   `json:"worker_id"`
	VideoID     string   `json:"video_id"`
	StartMs     int64    `json:"start_ms"`
	EndMs       int64    `json:"end_ms"`
	SegmentID   string   `json:"segment_id"`
	Source      string   `json:"source"`
	Observation string   `json:"observation"`
	Refs        []string `json:"refs"`
	Verified    bool     `json:"verified"`
	CreatedAt   int64    `json:"created_at"`
}

// Store wraps SQLite persistence for VideoRAG and the Evidence Store.
type Store struct {
	db *sql.DB
}

func Open(dbPath string) (*Store, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite %s: %w", dbPath, err)
	}
	if _, err := db.Exec(`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;`); err != nil {
		db.Close()
		return nil, fmt.Errorf("configure sqlite pragmas: %w", err)
	}
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error {
	if s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) migrate() error {
	const ddl = `
CREATE TABLE IF NOT EXISTS videos (
  video_id TEXT PRIMARY KEY,
  video_path TEXT NOT NULL,
  duration_ms INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  fps REAL NOT NULL,
  status TEXT NOT NULL,
  planner_status TEXT NOT NULL,
  planner_reason TEXT NOT NULL,
  build_visual_index INTEGER NOT NULL,
  stats_json TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
  video_id TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  asr TEXT NOT NULL,
  visual TEXT NOT NULL,
  speech_coverage REAL NOT NULL,
  visual_change REAL NOT NULL,
  visual_dependency INTEGER,
  description_completeness INTEGER,
  visual_status TEXT NOT NULL,
  sampled_timestamps_json TEXT NOT NULL,
  PRIMARY KEY (video_id, segment_id)
);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(video_id, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS vectors (
  video_id TEXT NOT NULL,
  channel TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  embedding BLOB NOT NULL,
  PRIMARY KEY (video_id, channel, segment_id)
);

CREATE TABLE IF NOT EXISTS evidence_store (
  evidence_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  worker_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  segment_id TEXT NOT NULL,
  source TEXT NOT NULL,
  observation TEXT NOT NULL,
  refs_json TEXT NOT NULL,
  verified INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence_store(run_id, video_id);
`
	_, err := s.db.Exec(ddl)
	if err != nil {
		return fmt.Errorf("migrate sqlite schema: %w", err)
	}
	return nil
}

func (s *Store) SaveVideo(v VideoRecord) error {
	if v.CreatedAt == 0 {
		v.CreatedAt = time.Now().UnixMilli()
	}
	bvi := 0
	if v.BuildVisualIndex {
		bvi = 1
	}
	_, err := s.db.Exec(`
INSERT INTO videos (video_id, video_path, duration_ms, width, height, fps, status, planner_status, planner_reason, build_visual_index, stats_json, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(video_id) DO UPDATE SET
  status=excluded.status,
  planner_status=excluded.planner_status,
  planner_reason=excluded.planner_reason,
  build_visual_index=excluded.build_visual_index,
  stats_json=excluded.stats_json`,
		v.VideoID, v.VideoPath, v.DurationMs, v.Width, v.Height, v.FPS,
		v.Status, v.PlannerStatus, v.PlannerReason, bvi, v.StatsJSON, v.CreatedAt,
	)
	return err
}

func (s *Store) GetVideo(videoID string) (*VideoRecord, error) {
	row := s.db.QueryRow(`
SELECT video_id, video_path, duration_ms, width, height, fps, status, planner_status, planner_reason, build_visual_index, stats_json, created_at
FROM videos WHERE video_id = ?`, videoID)
	var v VideoRecord
	var bvi int
	err := row.Scan(&v.VideoID, &v.VideoPath, &v.DurationMs, &v.Width, &v.Height, &v.FPS,
		&v.Status, &v.PlannerStatus, &v.PlannerReason, &bvi, &v.StatsJSON, &v.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	v.BuildVisualIndex = bvi != 0
	return &v, nil
}

func (s *Store) SaveSegments(segs []SegmentRecord) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(`
INSERT INTO segments (video_id, segment_id, start_ms, end_ms, asr, visual, speech_coverage, visual_change, visual_dependency, description_completeness, visual_status, sampled_timestamps_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(video_id, segment_id) DO UPDATE SET
  asr=excluded.asr,
  visual=excluded.visual,
  speech_coverage=excluded.speech_coverage,
  visual_change=excluded.visual_change,
  visual_dependency=excluded.visual_dependency,
  description_completeness=excluded.description_completeness,
  visual_status=excluded.visual_status,
  sampled_timestamps_json=excluded.sampled_timestamps_json`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, seg := range segs {
		tsJSON, _ := json.Marshal(seg.SampledTimestampsMs)
		if _, err := stmt.Exec(
			seg.VideoID, seg.SegmentID, seg.StartMs, seg.EndMs, seg.ASR, seg.Visual,
			seg.SpeechCoverage, seg.VisualChange, seg.VisualDependency, seg.DescriptionCompleteness,
			seg.VisualStatus, string(tsJSON),
		); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *Store) ListSegments(videoID string) ([]SegmentRecord, error) {
	rows, err := s.db.Query(`
SELECT video_id, segment_id, start_ms, end_ms, asr, visual, speech_coverage, visual_change, visual_dependency, description_completeness, visual_status, sampled_timestamps_json
FROM segments WHERE video_id = ? ORDER BY start_ms ASC`, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []SegmentRecord
	for rows.Next() {
		var seg SegmentRecord
		var tsJSON string
		if err := rows.Scan(
			&seg.VideoID, &seg.SegmentID, &seg.StartMs, &seg.EndMs, &seg.ASR, &seg.Visual,
			&seg.SpeechCoverage, &seg.VisualChange, &seg.VisualDependency, &seg.DescriptionCompleteness,
			&seg.VisualStatus, &tsJSON,
		); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(tsJSON), &seg.SampledTimestampsMs)
		out = append(out, seg)
	}
	return out, rows.Err()
}

func (s *Store) GetSegment(videoID, segmentID string) (*SegmentRecord, error) {
	row := s.db.QueryRow(`
SELECT video_id, segment_id, start_ms, end_ms, asr, visual, speech_coverage, visual_change, visual_dependency, description_completeness, visual_status, sampled_timestamps_json
FROM segments WHERE video_id = ? AND segment_id = ?`, videoID, segmentID)
	var seg SegmentRecord
	var tsJSON string
	err := row.Scan(
		&seg.VideoID, &seg.SegmentID, &seg.StartMs, &seg.EndMs, &seg.ASR, &seg.Visual,
		&seg.SpeechCoverage, &seg.VisualChange, &seg.VisualDependency, &seg.DescriptionCompleteness,
		&seg.VisualStatus, &tsJSON,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal([]byte(tsJSON), &seg.SampledTimestampsMs)
	return &seg, nil
}

func (s *Store) SaveVectors(vecs []VectorRecord) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(`
INSERT INTO vectors (video_id, channel, segment_id, model_id, embedding)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(video_id, channel, segment_id) DO UPDATE SET
  model_id=excluded.model_id,
  embedding=excluded.embedding`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, v := range vecs {
		if _, err := stmt.Exec(v.VideoID, v.Channel, v.SegmentID, v.ModelID, encodeFloat32s(v.Vector)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (s *Store) ListVectors(videoID, channel string) ([]VectorRecord, error) {
	rows, err := s.db.Query(`
SELECT video_id, channel, segment_id, model_id, embedding
FROM vectors WHERE video_id = ? AND channel = ?`, videoID, channel)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []VectorRecord
	for rows.Next() {
		var v VectorRecord
		var blob []byte
		if err := rows.Scan(&v.VideoID, &v.Channel, &v.SegmentID, &v.ModelID, &blob); err != nil {
			return nil, err
		}
		v.Vector = decodeFloat32s(blob)
		out = append(out, v)
	}
	return out, rows.Err()
}

func (s *Store) AddEvidence(ev EvidenceRecord) error {
	if ev.CreatedAt == 0 {
		ev.CreatedAt = time.Now().UnixMilli()
	}
	refsJSON, _ := json.Marshal(ev.Refs)
	verified := 0
	if ev.Verified {
		verified = 1
	}
	_, err := s.db.Exec(`
INSERT INTO evidence_store (evidence_id, run_id, worker_id, video_id, start_ms, end_ms, segment_id, source, observation, refs_json, verified, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(evidence_id) DO UPDATE SET
  observation=excluded.observation,
  refs_json=excluded.refs_json,
  verified=excluded.verified`,
		ev.EvidenceID, ev.RunID, ev.WorkerID, ev.VideoID, ev.StartMs, ev.EndMs,
		ev.SegmentID, ev.Source, ev.Observation, string(refsJSON), verified, ev.CreatedAt,
	)
	return err
}

func (s *Store) ListEvidence(runID, videoID string) ([]EvidenceRecord, error) {
	rows, err := s.db.Query(`
SELECT evidence_id, run_id, worker_id, video_id, start_ms, end_ms, segment_id, source, observation, refs_json, verified, created_at
FROM evidence_store WHERE run_id = ? AND video_id = ? ORDER BY created_at ASC`, runID, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []EvidenceRecord
	for rows.Next() {
		var ev EvidenceRecord
		var refsJSON string
		var verified int
		if err := rows.Scan(
			&ev.EvidenceID, &ev.RunID, &ev.WorkerID, &ev.VideoID, &ev.StartMs, &ev.EndMs,
			&ev.SegmentID, &ev.Source, &ev.Observation, &refsJSON, &verified, &ev.CreatedAt,
		); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(refsJSON), &ev.Refs)
		ev.Verified = verified != 0
		out = append(out, ev)
	}
	return out, rows.Err()
}

func encodeFloat32s(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

func decodeFloat32s(buf []byte) []float32 {
	n := len(buf) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return out
}
