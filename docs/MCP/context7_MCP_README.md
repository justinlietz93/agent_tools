# Context7 MCP Integration (Docker + HTTP)

Objective
- Use Context7 MCP as the single source for documentation lookup (no HTTP fallback to random sites)
- If the server is not running, auto-start a Docker container and try again
- If it still fails, return a clear message to the LLM: "Attempted to call Context7 twice and it was unavailable both times"

This repo wires Context7 MCP into the DocCheckTool strictly (no non-MCP fallback for "sites" checks) and provides a small client that auto-starts Docker when needed.

Repository Components
- Tool: `src/infrastructure/tools/mcp_tools/doc_check_tool.py` (strict MCP behavior for "sites")
- Client: `src/infrastructure/mcp/context7_client.py` (ensures server is running via Docker; HTTP health check)
- Dockerfile (optional): `docker/context7/Dockerfile` (simple server shell using `@upstash/context7-mcp`; not required if you use the official image)

Recommended: Use the official image
- Official image: `mcp/context7:latest`
- Transport: HTTP (SSE), defaults to port 8080
- Available tools via MCP:
  - `resolve-library-id`
  - `get-library-docs`

Quick Start (Official Image)
1) Ensure Docker daemon is running.

2) Run Context7 server:
   ```bash
   docker run -d --name context7-mcp \
     -p 8080:8080 \
     -e MCP_TRANSPORT=http \
     mcp/context7:latest
   ```

3) Configure environment for this project (optional; defaults used if omitted):
   - CONTEXT7_HTTP_URL=http://localhost:8080
   - CONTEXT7_CONTAINER_NAME=context7-mcp
   - CONTEXT7_DOCKER_IMAGE=mcp/context7:latest
   - CONTEXT7_PORT=8080
   - CONTEXT7_TIMEOUT_SECONDS=60

   Example `.env`:
   ```
   CONTEXT7_HTTP_URL=http://localhost:8080
   CONTEXT7_CONTAINER_NAME=context7-mcp
   CONTEXT7_DOCKER_IMAGE=mcp/context7:latest
   CONTEXT7_PORT=8080
   CONTEXT7_TIMEOUT_SECONDS=60
   ```

How it Works in Code
- DocCheckTool "sites" path calls:
  - `_check_sites_via_context7_strict(...)`
  - This uses `Context7MCPClient.ensure_running()` to:
    - Check if `http://localhost:8080` (or CONTEXT7_HTTP_URL) is reachable
    - If not: attempt to start the Docker container (create or start existing)
    - Wait up to CONTEXT7_TIMEOUT_SECONDS until server responds
  - If unable to reach the server twice, DocCheckTool returns:
    "Attempted to call Context7 twice and it was unavailable both times. Please start the Context7 MCP server ..."

Strict Behavior Rationale
- We intentionally do not add a fallback to direct HTTP checks of various random websites
- This keeps the documentation source consistent, reproducible, and centralized through Context7

Alternate: Use the local Dockerfile (optional)
A minimal Dockerfile exists in `docker/context7/Dockerfile`:
```dockerfile
FROM node:18-alpine
WORKDIR /app
RUN npm install -g @upstash/context7-mcp
# Default command to run the server
CMD ["context7-mcp"]
```
This starts a stdio-capable MCP server by default. Our client prefers HTTP transport (to enable health checks easily). For HTTP transport with this shell, you would need an HTTP wrapper that exposes the MCP server via SSE, which the official image already provides. Hence, prefer the official image.

Troubleshooting
- Container cannot start or is not present:
  - The client will attempt to create and start `context7-mcp` using `mcp/context7:latest`
  - Ensure you can pull the image: `docker pull mcp/context7:latest`
  - Ensure `docker version` works from your shell

- Port conflicts:
  - If port 8080 is in use, set `CONTEXT7_PORT` to a different host port and run:
    ```
    docker run -d --name context7-mcp -p 18080:8080 -e MCP_TRANSPORT=http mcp/context7:latest
    ```
  - Then set `CONTEXT7_HTTP_URL=http://localhost:18080`

- LLM message: "Attempted to call Context7 twice..."
  - Indicates the client could not reach the server and failed to (re)start it within the timeout
  - Verify Docker is running; inspect container logs: `docker logs context7-mcp`
  - Re-run the tool after resolving the issue

Notes and Next Steps
- The client currently ensures server availability and provides a foundation for adding real MCP tool calls (resolve-library-id, get-library-docs). If deeper integration is desired, implement those tool calls inside `Context7MCPClient` using the Python MCP client library over HTTP/SSE.
- For many workflows, only ensuring the MCP server is up is sufficient; LLM systems or editor plugins may perform the actual MCP tool invocations directly.
