package cloud

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// MultimodalEmbedder calls a cloud multimodal embedding endpoint (e.g. Qwen3-VL-Embedding / Jina-CLIP / OpenAI-compatible).
type MultimodalEmbedder struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

func NewMultimodalEmbedder(baseURL, apiKey, model string, timeout time.Duration) *MultimodalEmbedder {
	if baseURL == "" {
		baseURL = "https://api.siliconflow.cn/v1"
	}
	if model == "" {
		model = "Qwen/Qwen3-VL-Embedding-2B"
	}
	if timeout <= 0 {
		timeout = 90 * time.Second
	}
	return &MultimodalEmbedder{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Model:   model,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

func (m *MultimodalEmbedder) ModelID() string { return m.Model }

func (m *MultimodalEmbedder) EmbedQueries(ctx context.Context, queries []string) ([][]float32, error) {
	if len(queries) == 0 {
		return nil, nil
	}
	inputs := make([]map[string]string, len(queries))
	for i, q := range queries {
		inputs[i] = map[string]string{"text": q}
	}
	return m.call(ctx, inputs)
}

func (m *MultimodalEmbedder) EmbedImages(ctx context.Context, imagePaths []string) ([][]float32, error) {
	if len(imagePaths) == 0 {
		return nil, nil
	}
	inputs := make([]map[string]string, len(imagePaths))
	for i, p := range imagePaths {
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read image %s: %w", p, err)
		}
		inputs[i] = map[string]string{
			"image": "data:image/png;base64," + base64.StdEncoding.EncodeToString(data),
		}
	}
	return m.call(ctx, inputs)
}

func (m *MultimodalEmbedder) call(ctx context.Context, inputs []map[string]string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{
		"model": m.Model,
		"input": inputs,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+m.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := m.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("multimodal embedding HTTP %d: %s", resp.StatusCode, redact(string(raw), m.APIKey))
	}
	var parsed struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode multimodal embedding response: %w", err)
	}
	out := make([][]float32, len(inputs))
	for _, d := range parsed.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = d.Embedding
		}
	}
	return out, nil
}
