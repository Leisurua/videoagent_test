cmake_minimum_required(VERSION 3.20)
project(videoagent_media_daemon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat
  libavcodec
  libavutil
  libswscale
  libswresample
)
pkg_check_modules(LIBPNG REQUIRED IMPORTED_TARGET libpng)
pkg_check_modules(WHISPER REQUIRED IMPORTED_TARGET whisper)
pkg_check_modules(TESSERACT REQUIRED IMPORTED_TARGET tesseract)

set(PROTO_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

add_executable(media_daemon
  src/main.cpp
  src/ffmpeg_utils.cpp
  src/media_service_impl.cpp
  "${PROTO_GEN_DIR}/media_service.pb.cc"
  "${PROTO_GEN_DIR}/media_service.grpc.pb.cc"
)

target_include_directories(media_daemon PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${PROTO_GEN_DIR}"
)

target_link_libraries(media_daemon PRIVATE
  gRPC::grpc++
  gRPC::grpc++_reflection
  protobuf::libprotobuf
  PkgConfig::FFMPEG
  PkgConfig::LIBPNG
  PkgConfig::WHISPER
  PkgConfig::TESSERACT
)

if(WIN32)
  target_link_libraries(media_daemon PRIVATE ws2_32)
endif()
