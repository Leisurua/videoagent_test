#include "media_service_impl.hpp"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  std::string addr = "127.0.0.1:50051";
  if (argc > 1 && std::string(argv[1]) != "") {
    addr = argv[1];
  }

  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  videoagent::MediaServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "Failed to start gRPC MediaService on " << addr << std::endl;
    return 1;
  }

  std::cout << "videoagent media_daemon listening on " << addr << std::endl;
  server->Wait();
  return 0;
}
