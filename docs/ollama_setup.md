# Local Ollama Setup and Usage

This guide explains how to run Agent Tools with a local Ollama server via the OpenAI-compatible API.

## Prerequisites

- Python 3.8+
- Ollama installed: https://ollama.ai
- A running Ollama server

## 1) Install and start Ollama

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve
```

The default OpenAI-compatible base URL is:
- http://localhost:11434/v1

## 2) Pull a model

```bash
# Pull an LLM that supports tool-use/prompting well
ollama pull llama3.1
# or any other model you prefer:
# ollama pull qwen2.5-coder:7b
# ollama pull mistral
```

## 3) Install Python deps

```bash
pip install -r requirements.txt
```

## 4) Configure environment (optional)

Create `.env` in repo root to customize defaults:

```bash
# Ollama defaults (OpenAI-compatible)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1
# OLLAMA_API_KEY is typically unused; provided for compatibility
OLLAMA_API_KEY=ollama

# DeepSeek is optional now (no longer required)
# DEEPSEEK_API_KEY=your_key
```

Note: The framework no longer requires a DeepSeek API key for local usage.

## 5) Run the interactive CLI

```bash
# From repo root
python agent_tools/src/cli/app.py
```

Then choose:
- 2) Ollama (local)

Type natural language commands, e.g.:
- "Move the mouse to (500, 500)"
- "Type 'Hello World'"

Use `/tools` to list registered tools.

## 6) Programmatic usage

You may use the Ollama wrapper directly:

```python
from wrappers.ollama_wrapper import OllamaWrapper
from src.tools.tool_manager import ToolManager

wrapper = OllamaWrapper(  # Uses defaults from env or sensible fallbacks
    # base_url="http://localhost:11434/v1",
    # model="llama3.1",
)
manager = ToolManager()
for tool in manager.tools.values():
    wrapper.register_tool(tool)

result = wrapper.execute("Move the mouse to coordinates (100, 100)")
print(result)
```

## Troubleshooting

- "Connection refused":
  - Ensure `ollama serve` is running
  - Verify `OLLAMA_BASE_URL` matches your host/port
- "Model not found":
  - Run `ollama pull <model>` first
  - Set `OLLAMA_MODEL` in `.env` to a model you have pulled
- CLI shows no color:
  - Your terminal may not support ANSI colors
  - Try another terminal or disable color in your emulator