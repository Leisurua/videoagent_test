#include "media_service_impl.hpp"
#include "ffmpeg_utils.hpp"

#include <tesseract/baseapi.h>
#include <whisper.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace videoagent {

namespace fs = std::filesystem;

namespace {

struct AVFormatCloser {
  void operator()(AVFormatContext* ctx) const {
    if (ctx != nullptr) {
      avformat_close_input(&ctx);
    }
  }
};

using UniqueAVFormat = std::unique_ptr<AVFormatContext, AVFormatCloser>;

std::string av_err_str(int errnum) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errnum, buf, sizeof(buf));
  return std::string(buf);
}

std::string resolve_artifact_dir(const std::string& dir) {
  return dir.empty() ? "data/evidence" : dir;
}

std::string sanitize_id(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '-' || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "video" : out;
}

void fill_frame_artifact(
    videoagent::media::v1::FrameArtifact* out,
    const std::string& ref,
    const std::string& video_id,
    const std::string& image_path,
    int64_t requested_ms,
    const DecodedFrame& frame) {
  out->set_ref(ref);
  out->set_video_id(video_id);
  out->set_image_path(image_path);
  out->set_requested_ms(requested_ms);
  out->set_pts_ms(frame.pts_ms);
  out->set_display_end_ms(frame.display_end_ms);
  out->set_width(frame.width);
  out->set_height(frame.height);
}

grpc::Status map_exception(const std::exception& e, grpc::ServerContext* ctx) {
  if (ctx && ctx->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
  }
  return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
}

} // namespace

grpc::Status MediaServiceImpl::HealthCheck(
    grpc::ServerContext* /*context*/,
    const videoagent::media::v1::HealthCheckRequest* /*request*/,
    videoagent::media::v1::HealthCheckResponse* response) {
  response->set_ok(true);
  response->set_service_version("0.2.0");
  response->set_ffmpeg_version(av_version_info());
  response->set_whisper_available(true);
  response->set_tesseract_available(true);
  return grpc::Status::OK;
}

grpc::Status MediaServiceImpl::GetVideoMetadata(
    grpc::ServerContext* context,
    const videoagent::media::v1::GetVideoMetadataRequest* request,
    videoagent::media::v1::GetVideoMetadataResponse* response) {
  if (request->video_path().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "video_path is required");
  }
  if (context->IsCancelled()) {
    return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
  }

  AVFormatContext* raw_ctx = nullptr;
  int rc = avformat_open_input(&raw_ctx, request->video_path().c_str(), nullptr, nullptr);
  if (rc < 0) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        "failed to open video: " + av_err_str(rc));
  }
  UniqueAVFormat fmt_ctx(raw_ctx);

  rc = avformat_find_stream_info(fmt_ctx.get(), nullptr);
  if (rc < 0) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "failed to read stream info: " + av_err_str(rc));
  }

  int video_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_idx < 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "no video stream found in file");
  }

  AVStream* vstream = fmt_ctx->streams[video_idx];
  const AVCodecParameters* vpar = vstream->codecpar;
  response->set_width(vpar->width);
  response->set_height(vpar->height);

  if (const AVCodec* vcodec = avcodec_find_decoder(vpar->codec_id)) {
    response->set_video_codec(vcodec->name);
  }

  AVRational rate = av_guess_frame_rate(fmt_ctx.get(), vstream, nullptr);
  if (rate.num > 0 && rate.den > 0) {
    response->set_fps(av_q2d(rate));
  }

  if (fmt_ctx->duration != AV_NOPTS_VALUE && fmt_ctx->duration > 0) {
    response->set_duration_ms(fmt_ctx->duration / 1000);
  } else if (vstream->duration != AV_NOPTS_VALUE && vstream->duration > 0) {
    int64_t ms = av_rescale_q(vstream->duration, vstream->time_base, AVRational{1, 1000});
    response->set_duration_ms(ms);
  }

  int audio_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
  if (audio_idx >= 0) {
    response->set_has_audio(true);
    if (const AVCodec* acodec = avcodec_find_decoder(fmt_ctx->streams[audio_idx]->codecpar->codec_id)) {
      response->set_audio_codec(acodec->name);
    }
  } else {
    response->set_has_audio(false);
  }

  return grpc::Status::OK;
}

grpc::Status MediaServiceImpl::GetFrame(
    grpc::ServerContext* context,
    const videoagent::media::v1::GetFrameRequest* request,
    videoagent::media::v1::GetFrameResponse* response) {
  if (request->video_path().empty() || request->timestamp_ms() < 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "video_path and non-negative timestamp_ms are required");
  }
  try {
    auto cancel_cb = [context]() { return context->IsCancelled(); };
    DecodedFrame frame = decode_frame_at_ms(
        request->video_path(), request->timestamp_ms(), 0, 0, cancel_cb);

    fs::path dir(resolve_artifact_dir(request->artifact_dir()));
    fs::create_directories(dir);
    std::string vid = sanitize_id(request->video_id());
    std::string ref = vid + "_frame_" + std::to_string(frame.pts_ms);
    fs::path out_path = fs::absolute(dir / (ref + ".png"));
    write_png_rgb24(out_path.string(), frame.width, frame.height, frame.rgb24);

    fill_frame_artifact(
        response->mutable_frame(), ref, request->video_id(),
        out_path.string(), request->timestamp_ms(), frame);
    return grpc::Status::OK;
  } catch (const std::invalid_argument& e) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
  } catch (const std::exception& e) {
    return map_exception(e, context);
  }
}

grpc::Status MediaServiceImpl::SampleFrames(
    grpc::ServerContext* context,
    const videoagent::media::v1::SampleFramesRequest* request,
    videoagent::media::v1::SampleFramesResponse* response) {
  if (request->video_path().empty() || request->start_ms() < 0 ||
      request->end_ms() <= request->start_ms() || request->num_frames() <= 0 ||
      request->num_frames() > 32) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "invalid video_path, [start_ms, end_ms), or num_frames (1..32)");
  }

  try {
    int32_t n = request->num_frames();
    int64_t span = request->end_ms() - request->start_ms();
    std::vector<int64_t> targets;
    targets.reserve(n);
    for (int32_t i = 0; i < n; ++i) {
      // Uniform midpoints across n sub-intervals inside [start_ms, end_ms).
      int64_t ts = request->start_ms() + (span * (2 * i + 1)) / (2 * n);
      targets.push_back(ts);
    }

    auto cancel_cb = [context]() { return context->IsCancelled(); };
    auto frames = decode_frames_at_ms(
        request->video_path(), targets,
        request->target_width(), request->target_height(), cancel_cb);

    response->set_visual_change(compute_visual_change(frames));

    bool save_png = !request->artifact_dir().empty();
    fs::path dir;
    if (save_png) {
      dir = fs::path(request->artifact_dir());
      fs::create_directories(dir);
    }
    std::string vid = sanitize_id(request->video_id());

    for (size_t i = 0; i < frames.size(); ++i) {
      std::string ref = vid + "_sample_" + std::to_string(targets[i]) + "_" + std::to_string(i);
      std::string out_str;
      if (save_png) {
        fs::path out_path = fs::absolute(dir / (ref + ".png"));
        write_png_rgb24(out_path.string(), frames[i].width, frames[i].height, frames[i].rgb24);
        out_str = out_path.string();
      }
      fill_frame_artifact(
          response->add_frames(), ref, request->video_id(),
          out_str, targets[i], frames[i]);
    }
    return grpc::Status::OK;
  } catch (const std::invalid_argument& e) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
  } catch (const std::exception& e) {
    return map_exception(e, context);
  }
}

grpc::Status MediaServiceImpl::CropFrame(
    grpc::ServerContext* context,
    const videoagent::media::v1::CropFrameRequest* request,
    videoagent::media::v1::CropFrameResponse* response) {
  if (request->video_path().empty() || request->timestamp_ms() < 0 || !request->has_bbox()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "video_path, timestamp_ms, and bbox are required");
  }
  const auto& b = request->bbox();
  if (b.x1() < 0.0 || b.x1() >= b.x2() || b.x2() > 1.0 ||
      b.y1() < 0.0 || b.y1() >= b.y2() || b.y2() > 1.0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "bbox must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1");
  }

  try {
    auto cancel_cb = [context]() { return context->IsCancelled(); };
    DecodedFrame orig = decode_frame_at_ms(
        request->video_path(), request->timestamp_ms(), 0, 0, cancel_cb);

    int32_t px1 = std::max(0, static_cast<int32_t>(std::floor(b.x1() * orig.width)));
    int32_t py1 = std::max(0, static_cast<int32_t>(std::floor(b.y1() * orig.height)));
    int32_t px2 = std::min(orig.width, static_cast<int32_t>(std::ceil(b.x2() * orig.width)));
    int32_t py2 = std::min(orig.height, static_cast<int32_t>(std::ceil(b.y2() * orig.height)));
    int32_t cw = std::max(1, px2 - px1);
    int32_t ch = std::max(1, py2 - py1);

    auto cropped_rgb = crop_rgb24(orig.rgb24, orig.width, orig.height, px1, py1, cw, ch);

    fs::path dir(resolve_artifact_dir(request->artifact_dir()));
    fs::create_directories(dir);
    std::string vid = sanitize_id(request->video_id());

    std::string orig_ref = vid + "_frame_" + std::to_string(orig.pts_ms);
    fs::path orig_path = fs::absolute(dir / (orig_ref + ".png"));
    write_png_rgb24(orig_path.string(), orig.width, orig.height, orig.rgb24);

    std::string crop_ref = orig_ref + "_crop_" + std::to_string(px1) + "_" + std::to_string(py1) +
                           "_" + std::to_string(cw) + "x" + std::to_string(ch);
    fs::path crop_path = fs::absolute(dir / (crop_ref + ".png"));
    write_png_rgb24(crop_path.string(), cw, ch, cropped_rgb);

    fill_frame_artifact(
        response->mutable_original_frame(), orig_ref, request->video_id(),
        orig_path.string(), request->timestamp_ms(), orig);

    DecodedFrame crop_meta = orig;
    crop_meta.width = cw;
    crop_meta.height = ch;
    fill_frame_artifact(
        response->mutable_cropped_frame(), crop_ref, request->video_id(),
        crop_path.string(), request->timestamp_ms(), crop_meta);

    auto* rect = response->mutable_pixel_rect();
    rect->set_x(px1);
    rect->set_y(py1);
    rect->set_width(cw);
    rect->set_height(ch);
    return grpc::Status::OK;
  } catch (const std::invalid_argument& e) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
  } catch (const std::exception& e) {
    return map_exception(e, context);
  }
}

grpc::Status MediaServiceImpl::TranscribeASR(
    grpc::ServerContext* context,
    const videoagent::media::v1::TranscribeASRRequest* request,
    videoagent::media::v1::TranscribeASRResponse* response) {
  if (request->video_path().empty() || request->model_path().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "video_path and model_path (ggml whisper bin) are required");
  }
  if (!fs::exists(request->model_path())) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        "whisper model file not found: " + request->model_path());
  }

  try {
    auto cancel_cb = [context]() { return context->IsCancelled(); };
    std::vector<float> pcm = extract_audio_pcm_f32_16k(request->video_path(), cancel_cb);
    if (pcm.empty()) {
      return grpc::Status::OK;
    }

    whisper_context_params cparams = whisper_context_default_params();
    struct WhisperGuard {
      whisper_context* ctx = nullptr;
      ~WhisperGuard() {
        if (ctx) whisper_free(ctx);
      }
    } wguard{whisper_init_from_file_with_params(request->model_path().c_str(), cparams)};

    if (!wguard.ctx) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "failed to initialize whisper context");
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.token_timestamps = true;
    std::string lang = request->language().empty() ? "auto" : request->language();
    wparams.language = lang.c_str();
    wparams.abort_callback = [](void* user_data) -> bool {
      auto* ctx = static_cast<grpc::ServerContext*>(user_data);
      return ctx && ctx->IsCancelled();
    };
    wparams.abort_callback_user_data = context;

    if (whisper_full(wguard.ctx, wparams, pcm.data(), static_cast<int>(pcm.size())) != 0) {
      if (context->IsCancelled()) {
        return grpc::Status(grpc::StatusCode::CANCELLED, "ASR cancelled");
      }
      return grpc::Status(grpc::StatusCode::INTERNAL, "whisper_full failed");
    }

    int lang_id = whisper_full_lang_id(wguard.ctx);
    if (lang_id >= 0) {
      if (const char* lang_str = whisper_lang_str(lang_id)) {
        response->set_detected_language(lang_str);
      }
    }

    const int n_seg = whisper_full_n_segments(wguard.ctx);
    for (int i = 0; i < n_seg; ++i) {
      const char* txt = whisper_full_get_segment_text(wguard.ctx, i);
      if (!txt || txt[0] == '\0') continue;
      auto* span = response->add_spans();
      // whisper timestamps are in centiseconds (10ms units).
      span->set_start_ms(whisper_full_get_segment_t0(wguard.ctx, i) * 10);
      span->set_end_ms(whisper_full_get_segment_t1(wguard.ctx, i) * 10);
      span->set_text(txt);
    }
    return grpc::Status::OK;
  } catch (const std::exception& e) {
    return map_exception(e, context);
  }
}

grpc::Status MediaServiceImpl::OCRFrame(
    grpc::ServerContext* context,
    const videoagent::media::v1::OCRFrameRequest* request,
    videoagent::media::v1::OCRFrameResponse* response) {
  if (request->video_path().empty() || request->timestamp_ms() < 0) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "video_path and non-negative timestamp_ms are required");
  }

  try {
    auto cancel_cb = [context]() { return context->IsCancelled(); };
    DecodedFrame orig = decode_frame_at_ms(
        request->video_path(), request->timestamp_ms(), 0, 0, cancel_cb);

    int32_t offset_x = 0;
    int32_t offset_y = 0;
    int32_t ocr_w = orig.width;
    int32_t ocr_h = orig.height;
    std::vector<uint8_t> ocr_buf = orig.rgb24;

    bool has_crop = request->has_bbox() &&
                    (request->bbox().x2() > request->bbox().x1()) &&
                    (request->bbox().y2() > request->bbox().y1());
    if (has_crop) {
      const auto& b = request->bbox();
      if (b.x1() < 0.0 || b.x2() > 1.0 || b.y1() < 0.0 || b.y2() > 1.0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "bbox out of [0, 1]");
      }
      offset_x = std::max(0, static_cast<int32_t>(std::floor(b.x1() * orig.width)));
      offset_y = std::max(0, static_cast<int32_t>(std::floor(b.y1() * orig.height)));
      int32_t px2 = std::min(orig.width, static_cast<int32_t>(std::ceil(b.x2() * orig.width)));
      int32_t py2 = std::min(orig.height, static_cast<int32_t>(std::ceil(b.y2() * orig.height)));
      ocr_w = std::max(1, px2 - offset_x);
      ocr_h = std::max(1, py2 - offset_y);
      ocr_buf = crop_rgb24(orig.rgb24, orig.width, orig.height, offset_x, offset_y, ocr_w, ocr_h);
    }

    std::string vid = sanitize_id(request->video_id());
    std::string ref = vid + "_frame_" + std::to_string(orig.pts_ms);
    if (has_crop) {
      ref += "_crop_" + std::to_string(offset_x) + "_" + std::to_string(offset_y);
    }
    if (!request->artifact_dir().empty()) {
      fs::path dir(request->artifact_dir());
      fs::create_directories(dir);
      fs::path out_path = fs::absolute(dir / (ref + ".png"));
      write_png_rgb24(out_path.string(), ocr_w, ocr_h, ocr_buf);
    }
    response->set_image_ref(ref);

    tesseract::TessBaseAPI tess;
    std::string lang = request->language().empty() ? "eng" : request->language();
    if (tess.Init(nullptr, lang.c_str()) != 0) {
      return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                          "failed to initialize Tesseract for language: " + lang);
    }
    tess.SetImage(ocr_buf.data(), ocr_w, ocr_h, 3, ocr_w * 3);
    if (tess.Recognize(nullptr) != 0) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Tesseract Recognize failed");
    }

    std::unique_ptr<char[]> full(tess.GetUTF8Text());
    if (full) {
      response->set_full_text(full.get());
    }

    std::unique_ptr<tesseract::ResultIterator> ri(tess.GetIterator());
    if (ri) {
      do {
        if (context->IsCancelled()) {
          return grpc::Status(grpc::StatusCode::CANCELLED, "OCR cancelled");
        }
        std::unique_ptr<const char[]> word(ri->GetUTF8Text(tesseract::RIL_WORD));
        if (!word || word[0] == '\0') continue;
        float conf = ri->Confidence(tesseract::RIL_WORD);
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        if (!ri->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2)) continue;

        auto* w = response->add_words();
        w->set_text(word.get());
        w->set_confidence(conf >= 0.0f ? static_cast<double>(conf) / 100.0 : -1.0);
        auto* nb = w->mutable_bbox();
        nb->set_x1(static_cast<double>(offset_x + x1) / orig.width);
        nb->set_y1(static_cast<double>(offset_y + y1) / orig.height);
        nb->set_x2(static_cast<double>(offset_x + x2) / orig.width);
        nb->set_y2(static_cast<double>(offset_y + y2) / orig.height);
      } while (ri->Next(tesseract::RIL_WORD));
    }
    tess.End();
    return grpc::Status::OK;
  } catch (const std::invalid_argument& e) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());
  } catch (const std::exception& e) {
    return map_exception(e, context);
  }
}

} // namespace videoagent
