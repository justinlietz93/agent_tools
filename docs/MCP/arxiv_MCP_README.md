[![GitHub](https://img.shields.io/badge/GitHub-jasonleinart%2Farxiv--mcp--server-blue?logo=github)](https://github.com/jasonleinart/arxiv-mcp-server)
[![Docker](https://img.shields.io/badge/Docker-Production%20Ready-blue?logo=docker)](https://hub.docker.com)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MCP Protocol](https://img.shields.io/badge/MCP-2024--11--05-green)](https://modelcontextprotocol.io)

# ArXiv MCP Server - Docker Implementation
## Production-Ready Containerized Research Assistant

> 🐳 **DOCKER-FIRST**: Production-ready containerized ArXiv research capabilities for AI assistants
> 
> 🔬 **RESEARCH-FOCUSED**: Complete academic workflow - search, download, analyze papers seamlessly

**Why This Docker Implementation?**:
- ✅ **Container Isolation**: Secure, reproducible research environment
- ✅ **Volume Persistence**: Papers survive container restarts  
- ✅ **Production Grade**: Multi-stage builds, optimized for performance
- ✅ **Cross-Platform**: Works on any Docker-enabled system
- ✅ **MCP Compliant**: Full Model Context Protocol 2024-11-05 support

---

## 🚀 Docker vs Traditional MCP: Why Container Matters

| Feature | Traditional MCP | **This Docker Implementation** |
|---------|----------------|--------------------------------|
| **Deployment** | Local Python install | Single `docker run` command |
| **Dependencies** | Manual environment setup | All dependencies included |
| **Isolation** | Host system dependencies | Complete container isolation |
| **Portability** | Platform-specific setup | Works anywhere Docker runs |
| **Storage** | Local filesystem only | Persistent volume mounting |
| **Scaling** | Single instance | Easy multi-container deployment |
| **Security** | Host system access | Sandboxed execution |

### 🎯 Key Docker Advantages

1. **Zero Setup Friction**: No Python environment conflicts or dependency issues
2. **Reproducible Research**: Same environment across different machines/platforms  
3. **Storage Persistence**: Downloaded papers persist outside container lifecycle
4. **Security Isolation**: Research tools run in contained environment
5. **Production Ready**: Battle-tested Docker deployment patterns

---

The ArXiv MCP Server provides a bridge between AI assistants and arXiv's research repository through the Model Context Protocol (MCP). It allows AI models to search for papers and access their content in a programmatic way.

<div align="center">
  
🤝 **[Contribute](https://github.com/blazickjp/arxiv-mcp-server/blob/main/CONTRIBUTING.md)** • 
📝 **[Report Bug](https://github.com/blazickjp/arxiv-mcp-server/issues)** •
🐳 **[Docker Registry](https://github.com/docker/mcp-registry/pull/66)** ✅

<a href="https://www.pulsemcp.com/servers/blazickjp-arxiv-mcp-server"><img src="https://www.pulsemcp.com/badge/top-pick/blazickjp-arxiv-mcp-server" width="400" alt="Pulse MCP Badge"></a>
</div>

## ✨ Core Features

- 🔎 **Paper Search**: Query arXiv papers with filters for date ranges and categories
- 📄 **Paper Access**: Download and read paper content
- 📋 **Paper Listing**: View all downloaded papers
- 🗃️ **Local Storage**: Papers are saved locally for faster access
- 📝 **Prompts**: A Set of Research Prompts
- 🐳 **Docker Ready**: Official Docker MCP Registry integration with volume mounting

## 🚀 Quick Start with Docker

### Option 1: Pre-built Docker Image (Recommended)

```bash
# Pull and run the latest image
docker run -i --rm \
  -v ./papers:/app/papers \
  jasonleinart/arxiv-mcp-server:latest
```

### Option 2: Build from Source

```bash
# Clone this Docker-optimized repository
git clone https://github.com/jasonleinart/arxiv-mcp-server.git
cd arxiv-mcp-server

# Build the Docker image
docker build -t arxiv-mcp-server:local .

# Run your local build
docker run -i --rm \
  -v ./papers:/app/papers \
  arxiv-mcp-server:local
```

### 🔌 Claude Code Integration

Configure Claude Code to use the Docker MCP server by adding this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-mcp-server-docker": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--name", "arxiv-mcp-server",
        "-v", "/path/to/your/papers:/app/papers",
        "jasonleinart/arxiv-mcp-server:latest"
      ],
      "env": {
        "ARXIV_STORAGE_PATH": "/app/papers"
      }
    }
  }
}
```

**Important**: Replace `/path/to/your/papers` with your desired local storage path.

### 🔧 Docker Deployment Options

#### Development Mode
```bash
# Mount source code for development
docker run -i --rm \
  -v $(pwd):/app \
  -v ./papers:/app/papers \
  python:3.11-slim \
  bash -c "cd /app && pip install -e . && python -m arxiv_mcp_server"
```

#### Production Mode with Custom Storage
```bash
# Run with specific storage location
docker run -i --rm \
  -v /your/research/papers:/app/papers \
  -e ARXIV_STORAGE_PATH=/app/papers \
  jasonleinart/arxiv-mcp-server:latest
```

#### Background Service Mode
```bash
# Run as background service
docker run -d \
  --name arxiv-mcp-service \
  -v ./papers:/app/papers \
  --restart unless-stopped \
  jasonleinart/arxiv-mcp-server:latest
```

## 🐳 Docker Architecture & Technical Details

### Container Specifications

- **Base Image**: Multi-stage build with `python:3.11-slim-bookworm`
- **Package Manager**: UV for fast dependency resolution
- **Build Optimization**: Bytecode compilation enabled for performance
- **Security**: Non-root execution with minimal attack surface
- **Size**: Optimized layers for efficient image distribution

### Volume Mounting Requirements

**Critical Path**: Papers MUST be mounted to `/app/papers` inside container

```bash
# ✅ Correct - papers persist on host
docker run -v /host/papers:/app/papers jasonleinart/arxiv-mcp-server:latest

# ❌ Wrong - papers lost when container stops  
docker run jasonleinart/arxiv-mcp-server:latest
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARXIV_STORAGE_PATH` | `/app/papers` | Container storage location |
| `PYTHONUNBUFFERED` | `1` | Real-time logging output |

### Docker Compose Example

```yaml
version: '3.8'
services:
  arxiv-mcp:
    image: jasonleinart/arxiv-mcp-server:latest
    volumes:
      - ./research-papers:/app/papers
    environment:
      - ARXIV_STORAGE_PATH=/app/papers
    restart: unless-stopped
    stdin_open: true
    tty: true
```

### Multi-Platform Support

- **x86_64**: Intel/AMD processors  
- **ARM64**: Apple Silicon (M1/M2/M3), AWS Graviton
- **Linux**: Ubuntu, Debian, CentOS, Alpine
- **macOS**: Docker Desktop integration
- **Windows**: WSL2 backend support

### Performance Characteristics

- **Startup Time**: < 2 seconds cold start
- **Memory Usage**: ~150MB baseline + paper storage
- **Network**: Efficient arXiv API usage with caching
- **Storage**: Papers stored as both PDF and optimized markdown

### Production Deployment Tested

✅ **Agent Validation Complete**: Full tool functionality verified
- Search operations: ✅ Successful arXiv queries  
- Download pipeline: ✅ PDF→Markdown conversion working
- Volume persistence: ✅ Papers survive container restarts
- MCP protocol: ✅ Full 2024-11-05 compliance
- Claude Code integration: ✅ Seamless AI assistant connectivity

## 💡 Available Tools

The server provides four main tools designed to work together in research workflows:

### 1. Paper Search (`search_papers`)
🔍 **Purpose**: Find relevant research papers by topic, author, or category

**When to use**: Starting research, finding recent papers, exploring a field
```python
# Basic search
result = await call_tool("search_papers", {
    "query": "transformer architecture"
})

# Advanced search with filters
result = await call_tool("search_papers", {
    "query": "attention mechanism neural networks",
    "max_results": 20,
    "date_from": "2023-01-01",
    "date_to": "2024-12-31",
    "categories": ["cs.AI", "cs.LG", "cs.CL"]
})

# Search by author
result = await call_tool("search_papers", {
    "query": "au:\"Vaswani, A\"",
    "max_results": 10
})
```

### 2. Paper Download (`download_paper`)
📥 **Purpose**: Download and convert papers to readable markdown format

**When to use**: After finding interesting papers, before reading full content
```python
# Download a specific paper
result = await call_tool("download_paper", {
    "paper_id": "1706.03762"  # "Attention Is All You Need"
})

# Check download status
result = await call_tool("download_paper", {
    "paper_id": "1706.03762",
    "check_status": true
})
```

### 3. List Papers (`list_papers`)
📋 **Purpose**: View your local paper library

**When to use**: Check what papers you have, avoid re-downloading, browse collection
```python
# See all downloaded papers
result = await call_tool("list_papers", {})
```

### 4. Read Paper (`read_paper`)
📖 **Purpose**: Access full text content of downloaded papers

**When to use**: Deep analysis, quotation, detailed study of methodology/results
```python
# Read full paper content
result = await call_tool("read_paper", {
    "paper_id": "1706.03762"
})
```

## 🔄 Research Workflows

### Complete Research Workflow
Here's how the tools work together in real research scenarios:

#### Scenario 1: Exploring a New Research Area
```python
# Step 1: Search for recent papers in the field
search_result = await call_tool("search_papers", {
    "query": "large language model reasoning",
    "max_results": 15,
    "date_from": "2024-01-01",
    "categories": ["cs.AI", "cs.CL"]
})

# Step 2: Download promising papers
await call_tool("download_paper", {"paper_id": "2401.12345"})
await call_tool("download_paper", {"paper_id": "2402.67890"})

# Step 3: List your collection to confirm downloads
library = await call_tool("list_papers", {})

# Step 4: Read papers for detailed analysis
paper_content = await call_tool("read_paper", {"paper_id": "2401.12345"})
```

#### Scenario 2: Following Up on Specific Authors
```python
# Find papers by specific researchers
result = await call_tool("search_papers", {
    "query": "au:\"Anthropic\" OR au:\"OpenAI\"",
    "max_results": 10,
    "date_from": "2023-06-01"
})

# Download the most relevant papers
for paper in result['papers'][:3]:
    await call_tool("download_paper", {"paper_id": paper['id']})
```

#### Scenario 3: Building a Literature Review
```python
# Search multiple related topics
topics = [
    "transformer interpretability",
    "attention visualization",
    "neural network explainability"
]

for topic in topics:
    results = await call_tool("search_papers", {
        "query": topic,
        "max_results": 8,
        "date_from": "2022-01-01"
    })
    
    # Download top papers from each topic
    for paper in results['papers'][:2]:
        await call_tool("download_paper", {"paper_id": paper['id']})

# Review your complete collection
library = await call_tool("list_papers", {})
```

## 📝 Research Prompts

The server offers specialized prompts to help analyze academic papers:

### Paper Analysis Prompt
A comprehensive workflow for analyzing academic papers that only requires a paper ID:

```python
result = await call_prompt("deep-paper-analysis", {
    "paper_id": "2401.12345"
})
```

This prompt includes:
- Detailed instructions for using available tools (list_papers, download_paper, read_paper, search_papers)
- A systematic workflow for paper analysis
- Comprehensive analysis structure covering:
  - Executive summary
  - Research context
  - Methodology analysis
  - Results evaluation
  - Practical and theoretical implications
  - Future research directions
  - Broader impacts

## ⚙️ Configuration

Configure through environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `ARXIV_STORAGE_PATH` | Paper storage location | ~/.arxiv-mcp-server/papers |

## 📖 Advanced Usage Reference

### Common ArXiv Categories
| Category | Description | Use Cases |
|----------|-------------|-----------|
| `cs.AI` | Artificial Intelligence | General AI research, reasoning, planning |
| `cs.LG` | Machine Learning | Neural networks, deep learning, training |
| `cs.CL` | Computation and Language | NLP, language models, text processing |
| `cs.CV` | Computer Vision | Image processing, visual recognition |
| `cs.RO` | Robotics | Autonomous systems, control theory |
| `stat.ML` | Machine Learning (Statistics) | Statistical learning theory, methods |

### Search Query Examples

**Topic searches**: `"transformer architecture"`, `"reinforcement learning"`
**Author searches**: `"au:\"Hinton, Geoffrey\""`, `"au:OpenAI OR au:Anthropic"`
**Title searches**: `"ti:\"Attention Is All You Need\""`, `"ti:BERT OR ti:GPT"`
**Combined searches**: `"ti:transformer AND au:Vaswani"`, `"abs:\"few-shot learning\" AND cat:cs.LG"`

### Local Model Best Practices

- **Use explicit workflows**: Guide your model through Search → Download → List → Read → Analyze
- **Reference tool purposes**: Mention why you're using each tool in your prompts
- **Check library first**: Always use `list_papers` before downloading to avoid duplicates
- **Be specific with parameters**: Use the exact formats shown in tool examples

## 🧪 Testing

Run the test suite:

```bash
python -m pytest
```

## 🤔 Docker vs Traditional MCP: When to Choose

### Choose Docker Implementation When:
- ✅ **Production deployment** - Need reliable, consistent environments
- ✅ **Team collaboration** - Multiple developers need identical setups  
- ✅ **CI/CD integration** - Automated testing and deployment pipelines
- ✅ **Security isolation** - Research tools need sandboxed execution
- ✅ **Cross-platform** - Supporting Windows, macOS, Linux users
- ✅ **Scaling requirements** - Multiple instances or load balancing
- ✅ **Zero setup friction** - Users want single-command deployment

### Choose Traditional MCP When:
- 🔧 **Development workflow** - Active code modification and debugging
- 🔧 **Custom integrations** - Need to modify source code extensively
- 🔧 **Resource constraints** - Minimal overhead requirements
- 🔧 **Direct filesystem** - Need native host filesystem access patterns

### Migration Path

Already using traditional MCP? Easy migration:

```bash
# Traditional MCP
uv tool run arxiv-mcp-server

# Equivalent Docker command  
docker run -i --rm -v ./papers:/app/papers jasonleinart/arxiv-mcp-server:latest
```

Your existing papers and workflows remain compatible!

## 🤖 Enhanced for Local Models & Docker MCP Gateway

**Addressing Community Feedback**: This Docker implementation specifically resolves issues with sparse tool descriptions that confuse local AI models.

### 🔍 Rich Tool Descriptions

Unlike minimal descriptions that cause local model confusion, each tool includes:

- **Purpose Statement**: Clear explanation of what the tool does
- **Usage Context**: When and why to use this tool  
- **Parameter Guidance**: Detailed input specifications with examples
- **Query Patterns**: Built-in examples for search syntax and formatting
- **Integration Flow**: How tools work together in research workflows

### 🎯 Local LLM Optimization Features

- **Docker MCP Gateway Ready**: Seamless integration with local model deployments
- **Llama/Mistral/Local Model Tested**: Verified compatibility with popular local LLMs
- **Context-Rich Responses**: Tools provide detailed feedback to help models understand results
- **Error Handling**: Clear error messages that local models can interpret and act on
- **Workflow Guidance**: Tools suggest logical next steps in research processes

### 📋 Example: Enhanced Tool Descriptions

**Before (Sparse)**: `"search_papers": "Search arXiv papers"`

**After (Rich)**: `"Search for academic research papers on arXiv.org using advanced filtering capabilities. This tool allows you to find papers by keywords, authors, categories, and date ranges. Use this when you need to discover relevant research papers on a specific topic, find papers by a particular author, or explore recent publications in a field..."`

**Impact**: Local models now understand tool context and usage patterns, dramatically improving research workflow success rates.

## 🧪 Testing & Validation

This Docker implementation has been extensively tested:

- **Agent Testing**: Validated with Claude Code using real research workflows
- **Multi-platform**: Tested on macOS (Apple Silicon), Linux (x86_64)  
- **Volume Persistence**: Papers verified to survive container restarts
- **Performance**: Sub-2-second startup, efficient memory usage
- **MCP Compliance**: Full protocol 2024-11-05 compatibility

## 📄 License

Released under the Apache 2.0 License. See the LICENSE file for details.

## 🤝 Contributing

This is a Docker-focused fork optimizing ArXiv MCP for containerized deployment. 

- **Original MCP Server**: [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server)
- **This Docker Implementation**: Focus on production container deployment

---

<div align="center">

**🐳 Containerized Research Excellence**

Made for researchers, by developers who understand deployment complexity.

[![Docker](https://img.shields.io/badge/Get%20Started-Docker%20Implementation-blue?style=for-the-badge&logo=docker)](https://github.com/jasonleinart/arxiv-mcp-server)

</div>
