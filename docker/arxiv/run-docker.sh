docker run -d \
  --name arxiv-mcp-service \
  -v ./papers:/app/papers \
  --restart unless-stopped \
  mcp/arxiv-mcp-server:latest