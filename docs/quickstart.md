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
from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.computer_tool import ComputerTool

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