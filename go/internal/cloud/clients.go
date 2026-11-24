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

// TextEmbedder embeds text strings into dense vectors.
type TextEmbedder interface {
	ModelID() string
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// VisualEmbedder embeds images and text queries into a shared cross-modal space.
type VisualEmbedder interface {
	ModelID() string
	EmbedImages(ctx context.Context, imagePaths []string) ([][]float32, error)
	EmbedQueries(ctx context.Context, queries []string) ([][]float32, error)
}

// FrameInput describes one PNG artifact sent to a VLM.
type FrameInput struct {
	Ref       string `json:"ref"`
	ImagePath string `json:"image_path"`
	PTSMs     int64  `json:"pts_ms"`
}

// VisionRequest is the objective-observation contract for VLM calls.
type VisionRequest struct {
	Task     string       `json:"task"` // "inspect" or "locate_event"
	Question string       `json:"question"`
	StartMs  int64        `json:"start_ms"`
	EndMs    int64        `json:"end_ms"`
	Frames   []FrameInput `json:"frames"`
}

// VisionResult holds the VLM's objective observation without causal inference.
type VisionResult struct {
	Status       string   `json:"status"` // "matched", "no_content", "unreadable"
	Answer       string   `json:"answer"`
	EvidenceRefs []string `json:"evidence_refs"`
}

// VisionObserver observes supplied original frames without guessing unseen context.
type VisionObserver interface {
	ModelID() string
	Observe(ctx context.Context, req VisionRequest) (VisionResult, error)
}

// ToolCall represents an OpenAI-compatible function tool call.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ChatMessage is an OpenAI/DeepSeek-compatible conversation message.
type ChatMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content,omitempty"`
	ReasoningContent *string    `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// ToolDefinition describes a callable tool in OpenAI JSON Schema format.
type ToolDefinition struct {
	Type     string `json:"type"`
	Function struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  any    `json:"parameters"`
	} `json:"function"`
}

// TokenUsage tracks provider-reported token counts.
type TokenUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ChatRequest is passed to a ChatReasoner on each ReAct step.
type ChatRequest struct {
	Messages   []ChatMessage    `json:"messages"`
	Tools      []ToolDefinition `json:"tools,omitempty"`
	ToolChoice string           `json:"tool_choice,omitempty"`
}

// ChatResponse is the normalized assistant turn returned by a ChatReasoner.
type ChatResponse struct {
	Message ChatMessage `json:"message"`
	Usage   *TokenUsage `json:"usage,omitempty"`
}

// ChatReasoner is the main reasoning LLM interface.
type ChatReasoner interface {
	ModelID() string
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
}

// OpenAITextEmbedder calls an OpenAI-compatible /v1/embeddings endpoint.
type OpenAITextEmbedder struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

func NewOpenAITextEmbedder(baseURL, apiKey, model string, timeout time.Duration) *OpenAITextEmbedder {
	if baseURL == "" {
		baseURL = "https://api.siliconflow.cn/v1"
	}
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &OpenAITextEmbedder{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Model:   model,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

func (e *OpenAITextEmbedder) ModelID() string { return e.Model }

func (e *OpenAITextEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, _ := json.Marshal(map[string]any{
		"model": e.Model,
		"input": texts,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+e.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("embedding HTTP %d: %s", resp.StatusCode, redact(string(raw), e.APIKey))
	}
	var parsed struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode embedding response: %w", err)
	}
	out := make([][]float32, len(texts))
	for _, d := range parsed.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = d.Embedding
		}
	}
	return out, nil
}

// DeepSeekClient calls DeepSeek Chat Completions with thinking + tool calling enabled.
type DeepSeekClient struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

func NewDeepSeekClient(apiKey, model string, timeout time.Duration) *DeepSeekClient {
	if model == "" {
		model = "deepseek-chat"
	}
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	return &DeepSeekClient{
		BaseURL: "https://api.deepseek.com",
		APIKey:  apiKey,
		Model:   model,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

func (c *DeepSeekClient) ModelID() string { return c.Model }

func (c *DeepSeekClient) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	payload := map[string]any{
		"model":      c.Model,
		"messages":   req.Messages,
		"thinking":   map[string]string{"type": "enabled"},
		"max_tokens": 8192,
		"stream":     false,
	}
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
		if req.ToolChoice != "" {
			payload["tool_choice"] = req.ToolChoice
		}
	}
	body, _ := json.Marshal(payload)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return ChatResponse{}, fmt.Errorf("DeepSeek HTTP %d: %s", resp.StatusCode, redact(string(raw), c.APIKey))
	}

	var parsed struct {
		Choices []struct {
			FinishReason string      `json:"finish_reason"`
			Message      ChatMessage `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return ChatResponse{}, fmt.Errorf("decode DeepSeek response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return ChatResponse{}, fmt.Errorf("DeepSeek returned no choices")
	}
	if parsed.Choices[0].FinishReason == "length" {
		return ChatResponse{}, fmt.Errorf("DeepSeek output exceeded token limit")
	}
	return ChatResponse{
		Message: parsed.Choices[0].Message,
		Usage: &TokenUsage{
			InputTokens:  parsed.Usage.PromptTokens,
			OutputTokens: parsed.Usage.CompletionTokens,
		},
	}, nil
}

// SiliconFlowVision calls SiliconFlow's OpenAI-compatible Vision endpoint.
type SiliconFlowVision struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

func NewSiliconFlowVision(apiKey, model string, timeout time.Duration) *SiliconFlowVision {
	if model == "" {
		model = "Qwen/Qwen3.8-27B"
	}
	if timeout <= 0 {
		timeout = 90 * time.Second
	}
	return &SiliconFlowVision{
		BaseURL: "https://api.siliconflow.cn/v1",
		APIKey:  apiKey,
		Model:   model,
		HTTP:    &http.Client{Timeout: timeout},
	}
}

func (v *SiliconFlowVision) ModelID() string { return v.Model }

const objectiveVisionInstruction = `Answer only the caller's specific question using only the supplied visual frames. Do not treat assumptions in the question as facts. Do not infer intentions, causes, hidden actions, relationships or events outside these frames. If the frames provide no relevant visual information, return status "no_content" and answer "无相关内容". If relevant content is visible but unreadable or unrecognizable, return status "unreadable" and answer "无法辨认". Otherwise return status "matched" with a concise visible fact and cite supplied frame refs. Return one JSON object with exactly these keys: "status", "answer", "evidence_refs".`

func (v *SiliconFlowVision) Observe(ctx context.Context, req VisionRequest) (VisionResult, error) {
	content := make([]map[string]any, 0, len(req.Frames)*2+1)
	for _, f := range req.Frames {
		data, err := os.ReadFile(f.ImagePath)
		if err != nil {
			return VisionResult{}, fmt.Errorf("read frame %s: %w", f.ImagePath, err)
		}
		content = append(content, map[string]any{"type": "text", "text": fmt.Sprintf("Frame ref: %s (pts_ms: %d)", f.Ref, f.PTSMs)})
		content = append(content, map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url":    "data:image/png;base64," + base64.StdEncoding.EncodeToString(data),
				"detail": "high",
			},
		})
	}
	content = append(content, map[string]any{"type": "text", "text": req.Question})

	payload := map[string]any{
		"model":           v.Model,
		"stream":          false,
		"enable_thinking": false,
		"max_tokens":      1024,
		"response_format": map[string]string{"type": "json_object"},
		"messages": []map[string]any{
			{"role": "system", "content": objectiveVisionInstruction},
			{"role": "user", "content": content},
		},
	}
	body, _ := json.Marshal(payload)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, v.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return VisionResult{}, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+v.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := v.HTTP.Do(httpReq)
	if err != nil {
		return VisionResult{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return VisionResult{}, fmt.Errorf("SiliconFlow HTTP %d: %s", resp.StatusCode, redact(string(raw), v.APIKey))
	}

	var parsed struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return VisionResult{}, fmt.Errorf("decode SiliconFlow response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return VisionResult{}, fmt.Errorf("SiliconFlow returned no choices")
	}
	var out VisionResult
	if err := json.Unmarshal([]byte(parsed.Choices[0].Message.Content), &out); err != nil {
		return VisionResult{}, fmt.Errorf("parse vision JSON: %w", err)
	}
	return out, nil
}

func redact(s, secret string) string {
	if secret != "" {
		s = strings.ReplaceAll(s, secret, "[redacted]")
	}
	if len(s) > 500 {
		return s[:500]
	}
	return s
}
