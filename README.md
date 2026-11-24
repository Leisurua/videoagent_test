# VideoAgent (Go + C++ Architecture)

Super-Agent + 双通道 RAG + ReAct Harness 视频理解系统，采用 **Go（控制面 / Super-Agent Router / ReAct Harness / 云端 API 网关 / SQLite）+ C++（gRPC 媒体底座 / FFmpeg C API / whisper.cpp / libtesseract / libpng）** 架构。

## 目录结构

- `proto/media_service.proto`：Go 与 C++ 共享的 gRPC 契约（统一整数毫秒时间协议）
- `cpp/`：C++17 `media_daemon`（gRPC Server + FFmpeg + `libpng` + `whisper.cpp` + `libtesseract`）
- `go/`：
  - `internal/media`：gRPC 客户端与超时控制
  - `internal/store`：SQLite 持久化（`videos`、`segments`、`vectors` 与独立 `evidence_store`）
  - `internal/cloud`：云端 API 客户端（OpenAI 兼容 Embedding、DeepSeek Thinking + Tool Calling、SiliconFlow Qwen-VL）
  - `internal/rag`：30s 分片构建、双通道 RAG（文本 + 跨模态视觉）与 RRF 融合检索
  - `internal/tools`：9 个 ReAct 原子工具（`search_segments` / `get_segment` / `get_neighbors` / `inspect_segment` / `locate_event` / `get_frame` / `inspect_frame` / `ocr_frame` / `crop_frame`）
  - `internal/agent`：`ExecutionGuard`（125 步预算、65%/80%/90% 预警、去重、强制收尾）、`AgentRunner`、`read_tool_result` 分页与 Evidence 自动落盘
  - `internal/router`：Super-Agent Router（任务拆解、多 Worker 并发调度、共享预算、跨 Worker Evidence Store 聚合）
  - `cmd/videoagent`：完整 CLI 入口

## 构建与运行

### 1. 构建并启动 C++ `media_daemon`

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
$env:PKG_CONFIG_PATH = "C:\msys64\ucrt64\lib\pkgconfig"
cmake -S cpp -B cpp/build -G Ninja
cmake --build cpp/build
.\cpp\build\media_daemon.exe 127.0.0.1:50051
```

### 2. 测试与构建 Go CLI

```powershell
cd go
go test -v ./...
go build -o bin/videoagent.exe ./cmd/videoagent
```

### 3. 常用命令

```powershell
# 媒体服务健康检查与元信息探测
.\bin\videoagent.exe health
.\bin\videoagent.exe probe "D:\hsy258\videoagent\42934406924-1-192.mp4"

# 单帧提取 / 区间采样 / 局部裁剪
.\bin\videoagent.exe frame "D:\hsy258\videoagent\42934406924-1-192.mp4" 312840
.\bin\videoagent.exe sample "D:\hsy258\videoagent\42934406924-1-192.mp4" 300000 330000 3 320 180
.\bin\videoagent.exe crop "D:\hsy258\videoagent\42934406924-1-192.mp4" 312840 0.1 0.1 0.9 0.9

# 索引视频并运行 Super-Agent 问答（需设置 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY）
.\bin\videoagent.exe --asr-model path\to\ggml-small.bin index "D:\hsy258\videoagent\42934406924-1-192.mp4" my_video_01
.\bin\videoagent.exe ask my_video_01 "视频主要讲解什么？请附带时间点与画面依据。"

# 运行 A/B/C/D 四组消融实验评估
.\bin\videoagent.exe eval my_video_01 questions.json A,B,C,D
```

## 测试结果

参见 [Go/C++ 与云端模型测试报告](docs/qa/测试报告.md)，包括九道基准题的 A/B/C/D 消融、真实视频联调和已复现的问题。
