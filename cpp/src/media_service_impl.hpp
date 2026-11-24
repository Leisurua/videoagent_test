#pragma once

#include "media_service.grpc.pb.h"

namespace videoagent {

class MediaServiceImpl final : public videoagent::media::v1::MediaService::Service {
public:
  grpc::Status HealthCheck(
      grpc::ServerContext* context,
      const videoagent::media::v1::HealthCheckRequest* request,
      videoagent::media::v1::HealthCheckResponse* response) override;

  grpc::Status GetVideoMetadata(
      grpc::ServerContext* context,
      const videoagent::media::v1::GetVideoMetadataRequest* request,
      videoagent::media::v1::GetVideoMetadataResponse* response) override;

  grpc::Status TranscribeASR(
      grpc::ServerContext* context,
      const videoagent::media::v1::TranscribeASRRequest* request,
      videoagent::media::v1::TranscribeASRResponse* response) override;

  grpc::Status SampleFrames(
      grpc::ServerContext* context,
      const videoagent::media::v1::SampleFramesRequest* request,
      videoagent::media::v1::SampleFramesResponse* response) override;

  grpc::Status GetFrame(
      grpc::ServerContext* context,
      const videoagent::media::v1::GetFrameRequest* request,
      videoagent::media::v1::GetFrameResponse* response) override;

  grpc::Status CropFrame(
      grpc::ServerContext* context,
      const videoagent::media::v1::CropFrameRequest* request,
      videoagent::media::v1::CropFrameResponse* response) override;

  grpc::Status OCRFrame(
      grpc::ServerContext* context,
      const videoagent::media::v1::OCRFrameRequest* request,
      videoagent::media::v1::OCRFrameResponse* response) override;
};

} // namespace videoagent
