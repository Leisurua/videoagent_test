#include "ffmpeg_utils.hpp"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace videoagent {

namespace {

struct AVFormatCloser {
  void operator()(AVFormatContext* ctx) const {
    if (ctx) avformat_close_input(&ctx);
  }
};
struct AVCodecCloser {
  void operator()(AVCodecContext* ctx) const {
    if (ctx) avcodec_free_context(&ctx);
  }
};
struct AVFrameCloser {
  void operator()(AVFrame* f) const {
    if (f) av_frame_free(&f);
  }
};
struct AVPacketCloser {
  void operator()(AVPacket* p) const {
    if (p) av_packet_free(&p);
  }
};
struct SwsCloser {
  void operator()(SwsContext* s) const {
    if (s) sws_freeContext(s);
  }
};
struct SwrCloser {
  void operator()(SwrContext* s) const {
    if (s) swr_free(&s);
  }
};
struct FileCloser {
  void operator()(FILE* f) const {
    if (f) std::fclose(f);
  }
};

using UniqueFormat = std::unique_ptr<AVFormatContext, AVFormatCloser>;
using UniqueCodec = std::unique_ptr<AVCodecContext, AVCodecCloser>;
using UniqueFrame = std::unique_ptr<AVFrame, AVFrameCloser>;
using UniquePacket = std::unique_ptr<AVPacket, AVPacketCloser>;
using UniqueSws = std::unique_ptr<SwsContext, SwsCloser>;
using UniqueSwr = std::unique_ptr<SwrContext, SwrCloser>;
using UniqueFile = std::unique_ptr<FILE, FileCloser>;

std::string av_err_str(int errnum) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errnum, buf, sizeof(buf));
  return std::string(buf);
}

int64_t frame_pts_ms(const AVFrame* frame, AVRational time_base) {
  int64_t pts = frame->best_effort_timestamp;
  if (pts == AV_NOPTS_VALUE) {
    pts = frame->pts;
  }
  if (pts == AV_NOPTS_VALUE) {
    return 0;
  }
  return av_rescale_q(pts, time_base, AVRational{1, 1000});
}

std::vector<uint8_t> frame_to_rgb24(const AVFrame* src, int32_t out_w, int32_t out_h) {
  UniqueSws sws(sws_getContext(
      src->width, src->height, static_cast<AVPixelFormat>(src->format),
      out_w, out_h, AV_PIX_FMT_RGB24,
      SWS_BILINEAR, nullptr, nullptr, nullptr));
  if (!sws) {
    throw std::runtime_error("failed to create SwsContext");
  }
  std::vector<uint8_t> buf(static_cast<size_t>(out_w) * out_h * 3);
  uint8_t* dst_data[4] = {buf.data(), nullptr, nullptr, nullptr};
  int dst_linesize[4] = {out_w * 3, 0, 0, 0};
  sws_scale(sws.get(), src->data, src->linesize, 0, src->height, dst_data, dst_linesize);
  return buf;
}

DecodedFrame decode_single_in_open_ctx(
    AVFormatContext* fmt_ctx,
    int video_idx,
    AVCodecContext* codec_ctx,
    int64_t duration_ms,
    int64_t target_ms,
    int32_t target_w,
    int32_t target_h,
    const CancelCheck& is_cancelled) {
  AVStream* vstream = fmt_ctx->streams[video_idx];
  int64_t seek_ts = av_rescale_q(target_ms, AVRational{1, 1000}, vstream->time_base);
  if (av_seek_frame(fmt_ctx, video_idx, seek_ts, AVSEEK_FLAG_BACKWARD) < 0) {
    av_seek_frame(fmt_ctx, video_idx, 0, AVSEEK_FLAG_BACKWARD);
  }
  avcodec_flush_buffers(codec_ctx);

  UniquePacket pkt(av_packet_alloc());
  UniqueFrame cur_frame(av_frame_alloc());
  UniqueFrame best_frame(av_frame_alloc());
  bool has_best = false;
  int64_t best_pts_ms = 0;
  int64_t next_pts_ms = duration_ms > 0 ? duration_ms : target_ms;

  auto process_decoded = [&]() -> bool {
    while (true) {
      int rc = avcodec_receive_frame(codec_ctx, cur_frame.get());
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
        return false;
      }
      if (rc < 0) {
        throw std::runtime_error("avcodec_receive_frame failed: " + av_err_str(rc));
      }
      int64_t pts_ms = frame_pts_ms(cur_frame.get(), vstream->time_base);
      if (!has_best || pts_ms <= target_ms) {
        av_frame_unref(best_frame.get());
        av_frame_ref(best_frame.get(), cur_frame.get());
        has_best = true;
        best_pts_ms = pts_ms;
      } else {
        // First frame strictly after target_ms gives us display_end_ms.
        next_pts_ms = pts_ms;
        av_frame_unref(cur_frame.get());
        return true;
      }
      av_frame_unref(cur_frame.get());
    }
  };

  bool done = false;
  while (!done) {
    if (is_cancelled && is_cancelled()) {
      throw std::runtime_error("operation cancelled");
    }
    int rc = av_read_frame(fmt_ctx, pkt.get());
    if (rc == AVERROR_EOF) {
      avcodec_send_packet(codec_ctx, nullptr);
      done = process_decoded();
      break;
    }
    if (rc < 0) {
      throw std::runtime_error("av_read_frame failed: " + av_err_str(rc));
    }
    if (pkt->stream_index == video_idx) {
      if (avcodec_send_packet(codec_ctx, pkt.get()) >= 0) {
        done = process_decoded();
      }
    }
    av_packet_unref(pkt.get());
  }

  if (!has_best) {
    throw std::runtime_error("no video frame decoded near requested timestamp");
  }

  int32_t out_w = target_w > 0 ? target_w : best_frame->width;
  int32_t out_h = target_h > 0 ? target_h : best_frame->height;

  DecodedFrame result;
  result.width = out_w;
  result.height = out_h;
  result.pts_ms = best_pts_ms;
  result.display_end_ms = std::max(best_pts_ms, next_pts_ms);
  result.rgb24 = frame_to_rgb24(best_frame.get(), out_w, out_h);
  return result;
}

} // namespace

std::vector<DecodedFrame> decode_frames_at_ms(
    const std::string& video_path,
    const std::vector<int64_t>& target_ms_list,
    int32_t target_w,
    int32_t target_h,
    const CancelCheck& is_cancelled) {
  AVFormatContext* raw_fmt = nullptr;
  int rc = avformat_open_input(&raw_fmt, video_path.c_str(), nullptr, nullptr);
  if (rc < 0) {
    throw std::runtime_error("failed to open video: " + av_err_str(rc));
  }
  UniqueFormat fmt_ctx(raw_fmt);

  rc = avformat_find_stream_info(fmt_ctx.get(), nullptr);
  if (rc < 0) {
    throw std::runtime_error("failed to read stream info: " + av_err_str(rc));
  }

  int video_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_idx < 0) {
    throw std::runtime_error("no video stream found");
  }

  AVStream* vstream = fmt_ctx->streams[video_idx];
  const AVCodec* decoder = avcodec_find_decoder(vstream->codecpar->codec_id);
  if (!decoder) {
    throw std::runtime_error("unsupported video codec");
  }

  UniqueCodec codec_ctx(avcodec_alloc_context3(decoder));
  if (!codec_ctx) {
    throw std::runtime_error("failed to allocate codec context");
  }
  rc = avcodec_parameters_to_context(codec_ctx.get(), vstream->codecpar);
  if (rc < 0) {
    throw std::runtime_error("failed to copy codec parameters: " + av_err_str(rc));
  }
  rc = avcodec_open2(codec_ctx.get(), decoder, nullptr);
  if (rc < 0) {
    throw std::runtime_error("failed to open video decoder: " + av_err_str(rc));
  }

  int64_t duration_ms = 0;
  if (fmt_ctx->duration != AV_NOPTS_VALUE && fmt_ctx->duration > 0) {
    duration_ms = fmt_ctx->duration / 1000;
  }

  std::vector<DecodedFrame> out;
  out.reserve(target_ms_list.size());
  for (int64_t ts_ms : target_ms_list) {
    if (is_cancelled && is_cancelled()) {
      throw std::runtime_error("operation cancelled");
    }
    if (ts_ms < 0 || (duration_ms > 0 && ts_ms >= duration_ms)) {
      throw std::invalid_argument("timestamp_ms out of video duration range");
    }
    out.push_back(decode_single_in_open_ctx(
        fmt_ctx.get(), video_idx, codec_ctx.get(), duration_ms, ts_ms, target_w, target_h, is_cancelled));
  }
  return out;
}

DecodedFrame decode_frame_at_ms(
    const std::string& video_path,
    int64_t target_ms,
    int32_t target_w,
    int32_t target_h,
    const CancelCheck& is_cancelled) {
  auto frames = decode_frames_at_ms(video_path, {target_ms}, target_w, target_h, is_cancelled);
  return std::move(frames[0]);
}

void write_png_rgb24(
    const std::string& output_path,
    int32_t width,
    int32_t height,
    const std::vector<uint8_t>& rgb24) {
  UniqueFile fp(std::fopen(output_path.c_str(), "wb"));
  if (!fp) {
    throw std::runtime_error("failed to open PNG output path: " + output_path);
  }
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    throw std::runtime_error("png_create_write_struct failed");
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    throw std::runtime_error("png_create_info_struct failed");
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    throw std::runtime_error("libpng write error for: " + output_path);
  }

  png_init_io(png, fp.get());
  png_set_IHDR(
      png, info, static_cast<png_uint_32>(width), static_cast<png_uint_32>(height),
      8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);

  std::vector<png_bytep> rows(static_cast<size_t>(height));
  for (int32_t y = 0; y < height; ++y) {
    rows[y] = const_cast<png_bytep>(rgb24.data() + static_cast<size_t>(y) * width * 3);
  }
  png_write_image(png, rows.data());
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
}

std::vector<uint8_t> crop_rgb24(
    const std::vector<uint8_t>& src,
    int32_t src_w,
    int32_t /*src_h*/,
    int32_t x,
    int32_t y,
    int32_t w,
    int32_t h) {
  std::vector<uint8_t> dst(static_cast<size_t>(w) * h * 3);
  for (int32_t row = 0; row < h; ++row) {
    const uint8_t* src_row = src.data() + (static_cast<size_t>(y + row) * src_w + x) * 3;
    uint8_t* dst_row = dst.data() + static_cast<size_t>(row) * w * 3;
    std::copy(src_row, src_row + static_cast<size_t>(w) * 3, dst_row);
  }
  return dst;
}

double compute_visual_change(const std::vector<DecodedFrame>& frames) {
  if (frames.size() < 2) {
    return 0.0;
  }
  double total_diff = 0.0;
  size_t pairs = 0;
  for (size_t i = 1; i < frames.size(); ++i) {
    const auto& a = frames[i - 1].rgb24;
    const auto& b = frames[i].rgb24;
    if (a.size() != b.size() || a.empty()) {
      continue;
    }
    uint64_t sum = 0;
    for (size_t j = 0; j < a.size(); ++j) {
      sum += static_cast<uint64_t>(std::abs(static_cast<int>(a[j]) - static_cast<int>(b[j])));
    }
    total_diff += static_cast<double>(sum) / (static_cast<double>(a.size()) * 255.0);
    ++pairs;
  }
  return pairs > 0 ? total_diff / static_cast<double>(pairs) : 0.0;
}

std::vector<float> extract_audio_pcm_f32_16k(
    const std::string& video_path,
    const CancelCheck& is_cancelled) {
  AVFormatContext* raw_fmt = nullptr;
  int rc = avformat_open_input(&raw_fmt, video_path.c_str(), nullptr, nullptr);
  if (rc < 0) {
    throw std::runtime_error("failed to open audio source: " + av_err_str(rc));
  }
  UniqueFormat fmt_ctx(raw_fmt);

  rc = avformat_find_stream_info(fmt_ctx.get(), nullptr);
  if (rc < 0) {
    throw std::runtime_error("failed to read stream info: " + av_err_str(rc));
  }

  int audio_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
  if (audio_idx < 0) {
    return {};
  }

  AVStream* astream = fmt_ctx->streams[audio_idx];
  const AVCodec* decoder = avcodec_find_decoder(astream->codecpar->codec_id);
  if (!decoder) {
    throw std::runtime_error("unsupported audio codec");
  }

  UniqueCodec codec_ctx(avcodec_alloc_context3(decoder));
  if (!codec_ctx) {
    throw std::runtime_error("failed to allocate audio codec context");
  }
  if ((rc = avcodec_parameters_to_context(codec_ctx.get(), astream->codecpar)) < 0 ||
      (rc = avcodec_open2(codec_ctx.get(), decoder, nullptr)) < 0) {
    throw std::runtime_error("failed to open audio decoder: " + av_err_str(rc));
  }

  AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
  SwrContext* raw_swr = nullptr;
  rc = swr_alloc_set_opts2(
      &raw_swr,
      &out_layout, AV_SAMPLE_FMT_FLT, 16000,
      &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate,
      0, nullptr);
  if (rc < 0 || !raw_swr) {
    throw std::runtime_error("failed to configure SwrContext");
  }
  UniqueSwr swr(raw_swr);
  if ((rc = swr_init(swr.get())) < 0) {
    throw std::runtime_error("swr_init failed: " + av_err_str(rc));
  }

  std::vector<float> pcm;
  UniquePacket pkt(av_packet_alloc());
  UniqueFrame frame(av_frame_alloc());

  auto drain_frames = [&]() {
    while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
      int out_samples = swr_get_out_samples(swr.get(), frame->nb_samples);
      if (out_samples > 0) {
        size_t old_size = pcm.size();
        pcm.resize(old_size + static_cast<size_t>(out_samples));
        uint8_t* out_ptr = reinterpret_cast<uint8_t*>(pcm.data() + old_size);
        int converted = swr_convert(
            swr.get(), &out_ptr, out_samples,
            const_cast<const uint8_t**>(frame->extended_data), frame->nb_samples);
        if (converted < 0) {
          throw std::runtime_error("swr_convert failed");
        }
        pcm.resize(old_size + static_cast<size_t>(converted));
      }
      av_frame_unref(frame.get());
    }
  };

  while (true) {
    if (is_cancelled && is_cancelled()) {
      throw std::runtime_error("operation cancelled");
    }
    rc = av_read_frame(fmt_ctx.get(), pkt.get());
    if (rc == AVERROR_EOF) {
      avcodec_send_packet(codec_ctx.get(), nullptr);
      drain_frames();
      break;
    }
    if (rc < 0) {
      throw std::runtime_error("av_read_frame audio failed: " + av_err_str(rc));
    }
    if (pkt->stream_index == audio_idx) {
      if (avcodec_send_packet(codec_ctx.get(), pkt.get()) >= 0) {
        drain_frames();
      }
    }
    av_packet_unref(pkt.get());
  }

  // Flush resampler delay buffer.
  int rem = swr_get_out_samples(swr.get(), 0);
  if (rem > 0) {
    size_t old_size = pcm.size();
    pcm.resize(old_size + static_cast<size_t>(rem));
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(pcm.data() + old_size);
    int converted = swr_convert(swr.get(), &out_ptr, rem, nullptr, 0);
    pcm.resize(old_size + static_cast<size_t>(std::max(0, converted)));
  }

  return pcm;
}

} // namespace videoagent
