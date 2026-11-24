#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace videoagent {

struct DecodedFrame {
  int32_t width = 0;
  int32_t height = 0;
  int64_t pts_ms = 0;
  int64_t display_end_ms = 0;
  std::vector<uint8_t> rgb24;
};

using CancelCheck = std::function<bool()>;

// Decodes the frame currently displayed at target_ms (integer milliseconds).
// If target_w > 0 and target_h > 0, scales the output RGB24 buffer to that size.
// Throws std::runtime_error on media errors or cancellation.
DecodedFrame decode_frame_at_ms(
    const std::string& video_path,
    int64_t target_ms,
    int32_t target_w,
    int32_t target_h,
    const CancelCheck& is_cancelled);

// Decodes multiple frames in a single open container pass.
std::vector<DecodedFrame> decode_frames_at_ms(
    const std::string& video_path,
    const std::vector<int64_t>& target_ms_list,
    int32_t target_w,
    int32_t target_h,
    const CancelCheck& is_cancelled);

// Writes an RGB24 buffer to a PNG file using libpng.
void write_png_rgb24(
    const std::string& output_path,
    int32_t width,
    int32_t height,
    const std::vector<uint8_t>& rgb24);

// Crops an RGB24 buffer to [x, x+w) x [y, y+h).
std::vector<uint8_t> crop_rgb24(
    const std::vector<uint8_t>& src,
    int32_t src_w,
    int32_t src_h,
    int32_t x,
    int32_t y,
    int32_t w,
    int32_t h);

// Computes mean absolute RGB difference / 255 across adjacent frames in [0, 1].
double compute_visual_change(const std::vector<DecodedFrame>& frames);

// Extracts audio from video_path into 16kHz mono float32 PCM for whisper.cpp.
std::vector<float> extract_audio_pcm_f32_16k(
    const std::string& video_path,
    const CancelCheck& is_cancelled);

} // namespace videoagent
