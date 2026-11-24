package media

import (
	"context"
	"fmt"
	"time"

	mediapb "github.com/videoagent/videoagent_project/go/internal/media/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client wraps the gRPC connection to the C++ media_daemon.
type Client struct {
	conn *grpc.ClientConn
	rpc  mediapb.MediaServiceClient
}

// Dial connects to the C++ media_daemon at addr (e.g., "127.0.0.1:50051").
func Dial(ctx context.Context, addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial media_daemon %s: %w", addr, err)
	}
	return &Client{
		conn: conn,
		rpc:  mediapb.NewMediaServiceClient(conn),
	}, nil
}

func (c *Client) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *Client) HealthCheck(ctx context.Context) (*mediapb.HealthCheckResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return c.rpc.HealthCheck(ctx, &mediapb.HealthCheckRequest{})
}

func (c *Client) GetVideoMetadata(ctx context.Context, videoPath string) (*mediapb.GetVideoMetadataResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	return c.rpc.GetVideoMetadata(ctx, &mediapb.GetVideoMetadataRequest{
		VideoPath: videoPath,
	})
}

func (c *Client) GetFrame(ctx context.Context, req *mediapb.GetFrameRequest) (*mediapb.GetFrameResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	return c.rpc.GetFrame(ctx, req)
}

func (c *Client) SampleFrames(ctx context.Context, req *mediapb.SampleFramesRequest) (*mediapb.SampleFramesResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	return c.rpc.SampleFrames(ctx, req)
}

func (c *Client) CropFrame(ctx context.Context, req *mediapb.CropFrameRequest) (*mediapb.CropFrameResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	return c.rpc.CropFrame(ctx, req)
}

func (c *Client) TranscribeASR(ctx context.Context, req *mediapb.TranscribeASRRequest) (*mediapb.TranscribeASRResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	return c.rpc.TranscribeASR(ctx, req)
}

func (c *Client) OCRFrame(ctx context.Context, req *mediapb.OCRFrameRequest) (*mediapb.OCRFrameResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	return c.rpc.OCRFrame(ctx, req)
}
