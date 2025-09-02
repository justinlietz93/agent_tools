# Deepseek Reasoner Quickstart Guide

A concise guide to get Deepseek Reasoner up and running with agent tools.

## System Requirements

- Python 3.8+
- Windows OS
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/justinlietz93/agent_tools.git
cd agent_tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API access:
   - Get your API key from [platform.deepseek.com](https://platform.deepseek.com)
   - Create `.env` file in project root:
   ```bash
   DEEPSEEK_API_KEY=your_api_key_here
   ```

## Verification

Run the integration test:
```bash
pytest -v -s tests/test_deepseek_tools.py::test_real_notepad_demo -m real
```

This will demonstrate Deepseek controlling your system by:
1. Opening Notepad
2. Typing a test message

## Basic Usage

Deepseek Reasoner understands natural language commands and automatically determines the appropriate tool actions. Simply describe what you want to do:

```python
from src.infrastructure.llm.deepseek_wrapper import DeepseekToolWrapper
from src.infrastructure.tools.computer_tool import ComputerTool

# Initialize
wrapper = DeepseekToolWrapper()
computer = ComputerTool()
wrapper.register_tool(computer)

# Natural language commands - Deepseek figures out the right actions
wrapper.execute("Move the mouse to the top-left corner")
wrapper.execute("Double click")
wrapper.execute("Type 'This is automatic!'")
```

## Common Operations

Deepseek understands various ways of expressing commands:

### Mouse Control
```python
# All these do the same thing
wrapper.execute("Move mouse to (500, 500)")
wrapper.execute("Put the cursor at coordinates 500, 500")
wrapper.execute("Move pointer to x=500, y=500")

# Clicking
wrapper.execute("Click the left mouse button")
wrapper.execute("Double click")
wrapper.execute("Right click here")
```

### Keyboard Input
```python
# Natural text input
wrapper.execute("Type 'Any text you want'")
wrapper.execute("Write 'It understands context'")

# Key combinations
wrapper.execute("Press Ctrl+C")
wrapper.execute("Hit the Enter key")
wrapper.execute("Press Windows key")
```

### Window Management
```python
# Finding and focusing windows
wrapper.execute("Focus the Chrome window")
wrapper.execute("Switch to Notepad")
wrapper.execute("Find the calculator window")

# Window positioning
wrapper.execute("Move the current window to the top-left")
wrapper.execute("Center this window on screen")
```

## Advanced Example: AI Code Generation

This example demonstrates Deepseek's powerful reasoning capabilities when generating and running code:

```python
# Generate a Pomodoro timer application
result = wrapper.execute(
    """Create a Python program that implements a Pomodoro timer with:
    1. GUI using tkinter
    2. 25-minute work timer
    3. Desktop notifications
    4. Progress bar
    [etc...]"""
)
```

### Deepseek's Chain of Thought

When asked to run the generated program, Deepseek shows impressive reasoning:

```
Reasoning:
1. First, I need to figure out how to execute the script. Looking at available tools:
   - 'file' tool: handles file operations
   - 'computer' tool: controls mouse, keyboard, windows

2. To run a Python script, typically you would:
   - Open terminal
   - Type 'python script.py'
   - Press Enter

3. Analyzing possible approaches:
   - Could use keyboard shortcuts to open terminal (Win+R, cmd)
   - Could assume terminal is already open
   - Need to ensure correct working directory

4. Considering tool capabilities:
   - 'type' action can input text
   - 'key' action can send key combinations
   - Need to handle newlines and Enter key

5. Solution:
   Use 'type' action with newline to simulate command execution:
   python examples/pomodoro_timer.py\n
```

This demonstrates how Deepseek:
- Analyzes available tools
- Considers multiple approaches
- Handles edge cases
- Makes reasonable assumptions
- Explains its decision process

## Troubleshooting

### Common Issues

#### Windows API Error: "EnumWindows: No error message is available"
This usually means the Windows API bindings aren't properly registered. To fix:

```bash
# 1. Uninstall pywin32
pip uninstall pywin32

# 2. Reinstall pywin32
pip install pywin32

# 3. Run the post-install script as administrator
python C:\Users\YourUsername\AppData\Local\Programs\Python\Python3x\Scripts\pywin32_postinstall.py -install
```

Note: Replace `YourUsername` and `Python3x` with your actual username and Python version.

If you encounter:

`ModuleNotFoundError`:
```bash
python -m pip install -r requirements.txt
```

`Invalid API Key`:
- Verify key in .env
- Check platform.deepseek.com permissions

`Permission Error`:
```bash
# Run terminal as administrator
```

## Next Steps

- [Tool Development Guide](tool_development.md)
- [API Documentation](https://api-docs.deepseek.com/)
- [Example Scripts](/examples)
# Context7 MCP DocCheckTool (First-class)

This project includes a Context7-only documentation check tool that fetches third‑party docs from the Context7 MCP server. It does not read your local files. It validates provider-returned documentation, not your codebase.

Enablement

- Start Context7 MCP (HTTP/SSE):
  - bun: bun run dist/index.js --transport http --port 8080
  - Docker (recommended): docker run -d --name context7-mcp -p 8080:8080 -e MCP_TRANSPORT=http mcp/context7:latest
- Auto-enable vs. opt-in:
  - Auto: If the server is reachable on startup, the tool registers automatically.
  - Opt-in: export CONTEXT7_ENABLE_DOC_TOOL=1
- Base URL override: export CONTEXT7_BASE_URL=http://localhost:8080

Standardized output

All calls return a dict:
- status: "success" | "error"
- check_type: "completeness" | "sites" | "links" | "anchors" | "frontmatter"
- summary: Human-readable
- data: Provider-native payload and/or analysis
- base_url: Context7 endpoint used

Usage examples (/call)

- Completeness (library docs)
  {
    "tool": "documentation_check",
    "input_schema": {
      "check_type": "completeness",
      "library": "/upstash/context7",
      "required_sections": ["Introduction", "API Reference"],
      "tokens": 2000
    }
  }

- Sites (multiple libraries)
  {
    "tool": "documentation_check",
    "input_schema": {
      "check_type": "sites",
      "libraries": ["react", "vue", "/upstash/context7"],
      "tokens": 1000
    }
  }

Notes
- links/anchors/frontmatter are currently not supported by Context7 in this adapter. Calls return a structured "error"/"unsupported" with a clear summary.
- The older DocCheckTool that scanned local files has been replaced. The new tool uses Context7 only and never falls back to LLMs or local parsing.

Tests
- Anthropic-style LLM tests are skipped by default during collection. Enable them explicitly:
  export RUN_ANTHROPIC_TESTS=1
  pytest -q