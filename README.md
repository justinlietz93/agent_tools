# Agent Tools

A Python framework for building AI agent tools that can interact with various systems and APIs.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/justinlietz93/agent_tools.git
cd agent_tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your API keys
ANTHROPIC_API_KEY=your_api_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## Using Tools with Deepseek Reasoner

The framework uses Deepseek's Reasoner model to control tools through prompt engineering. Deepseek Reasoner provides Chain of Thought (CoT) reasoning before executing actions.

### Compatibility

All tools built for this framework work automatically with Deepseek Reasoner - no additional implementation needed! The DeepseekToolWrapper handles:
- Converting tool schemas to Deepseek-friendly formats
- Managing the Chain of Thought reasoning
- Parsing tool calls from responses
- Error handling and response formatting

### Important Implementation Notes

1. Tool Output Format
Your tool's `run()` method must return a dictionary with this structure:
```python
{
    "type": "tool_result",  # Required
    "content": {
        "status": "success" | "error",  # Required
        "message": "Description of what happened",  # Required
        # Any additional data specific to your tool
    }
}
```

2. Tool Schema Requirements
- All input_schema must have clear descriptions
- Parameter types must be explicitly defined
- Required input_schema must be listed

Example of a properly formatted tool:
```python
from tools.tool_base import Tool

class YourTool(Tool):
    name = "your_tool"
    description = "Description of what your tool does"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform",
                "enum": ["action1", "action2"]  # List allowed values
            },
            "value": {
                "type": "number",
                "description": "The value to use"
            }
        },
        "required": ["action", "value"]
    }

    def run(self, params):
        try:
            # Your implementation
            return {
                "type": "tool_result",
                "content": {
                    "status": "success",
                    "message": "Action completed",
                    "data": result_data
                }
            }
        except Exception as e:
            return {
                "type": "tool_result",
                "content": {
                    "status": "error",
                    "message": str(e)
                }
            }
```

### Basic Usage

```python
from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.computer_tool import ComputerTool

# Initialize wrapper and tools
wrapper = DeepseekToolWrapper()
computer = ComputerTool()
wrapper.register_tool(computer)

# Execute a tool based on user input
result = wrapper.execute("Move the mouse to coordinates (100, 100)")

# Result includes:
# - Reasoning from Deepseek
# - Tool call details
# - Tool execution result
```

### Tool Registration

Tools must implement the base Tool interface and provide:
- A name
- A description
- An input schema
- A run method

```python
from tools.tool_base import Tool

class YourTool(Tool):
    name = "your_tool"
    description = "Description of what your tool does"
    input_schema = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of param1"
            }
        },
        "required": ["param1"]
    }

    def run(self, params):
        # Tool implementation
        pass
```

### Understanding Responses

When you execute a tool, you'll get a response with three parts:

1. Reasoning: Deepseek's Chain of Thought explanation
2. Tool Call: The structured command to be executed
3. Result: The output from the tool execution

Example response:
```
Reasoning:
To move the mouse to coordinates (100, 100), I'll use the computer tool's mouse_move action.
This is within safe screen boundaries and will use a smooth movement.

Tool Call:
{
    "tool": "computer",
    "input_schema": {
        "action": "mouse_move",
        "coordinate": [100, 100]
    }
}

Result:
{
    "status": "success",
    "message": "Mouse moved to (100, 100)"
}
```

## Running Tests

The test suite is organized with different markers for various test types:

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run LLM-specific tests
pytest -m llm

# Run real system tests
pytest -m real

# Run demo tests
pytest -m demo
```

### Test Categories

- `unit`: Tests with mocked dependencies
- `integration`: Tests that check component integration
- `llm`: Tests specifically for LLM interactions
- `real`: Tests using real system interactions
- `demo`: Visual demonstration tests
- `real_http`: Tests using real HTTP endpoints
- `system`: Full system tests with real LLM and targets
- `slow`: Tests that take longer to run

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

PLEASE feel free to submit issues or feature requests in the issues section.

## Currently in queue
- Dynamic LLM web search (natural language prompting the LLM to find anything on the internet)
- LLM web scraping and data aggregation. (Doc building, research, etc)
- Add functionality to allow any LLM to pilot the system, including self-hosted
- Lastly, I have a huge update coming to allow fully autonomous, persistent memory, goal pursuit LLM capabilities with many more tools. The LLM will be capable of building and registering it's own tools at runtime and still be compatible with the previous features. I have been testing this and proven it works, but it is not stable enough to release. Stay tuned

## License

MIT License

Copyright (c) 2024 Justin Lietz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
