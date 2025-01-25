# Setting Up Deepseek Reasoner with Agent Tools

This guide will walk you through setting up Deepseek Reasoner to use the agent tools framework.

## Prerequisites

1. Python 3.8 or higher
2. A Deepseek API key ([Get one here](https://platform.deepseek.com))
3. The agent_tools repository

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in your project root:
```bash
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. Basic Usage Example

Here's a simple example using the computer tool:

```python
from tools.deepseek_wrapper import DeepseekToolWrapper
from tools.computer_tool import ComputerTool

# Initialize the wrapper
wrapper = DeepseekToolWrapper()

# Create and register your tool
computer = ComputerTool()
wrapper.register_tool(computer)

# Use the tool through natural language
result = wrapper.execute("Move the mouse to coordinates (100, 100)")
print(result)
```

### 4. Understanding the Response

The response includes three parts:
```python
# Example response structure
{
    "reasoning": "I'll use the computer tool to move the mouse...",
    "tool_call": {
        "tool": "computer",
        "input_schema": {
            "action": "mouse_move",
            "coordinate": [100, 100]
        }
    },
    "result": {
        "status": "success",
        "message": "Mouse moved to (100, 100)"
    }
}
```

### 5. Using Multiple Tools

You can register multiple tools:

```python
# Register multiple tools
wrapper.register_tool(computer_tool)
wrapper.register_tool(web_tool)
wrapper.register_tool(api_tool)

# Deepseek will automatically choose the appropriate tool
result = wrapper.execute("Open Chrome and navigate to google.com")
```

### 6. Error Handling

The wrapper automatically handles errors:
```python
try:
    result = wrapper.execute("Invalid command")
except Exception as e:
    print(f"Error: {e}")
```

## Advanced Usage

### Chain of Thought Reasoning

Deepseek Reasoner provides detailed reasoning before executing actions. You can access this:

```python
result = wrapper.execute("Complex task description")
print("Reasoning:", result.split("Reasoning:\n")[1].split("\n\nTool Call:")[0])
```

### Customizing Tool Behavior

The wrapper automatically converts your tool's schema into natural language instructions. Make sure your tool has:

1. Clear descriptions
2. Well-defined parameter types
3. Helpful error messages

Example:
```python
class YourTool(Tool):
    name = "your_tool"
    description = "A clear description of what your tool does"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Very clear description of this parameter",
                "enum": ["action1", "action2"]
            }
        },
        "required": ["action"]
    }
```

## Testing

Test your Deepseek integration:

```bash
# Run Deepseek-specific tests
pytest -m llm

# Run integration tests
pytest -m integration
```

## Troubleshooting

Common issues and solutions:

1. **Invalid API Key**
   - Ensure your API key is correctly set in .env
   - Check API key permissions

2. **Tool Registration Fails**
   - Verify tool implements required interface
   - Check schema format

3. **Execution Errors**
   - Look for missing input_schema
   - Check parameter types match schema

## Additional Resources

- [Deepseek API Documentation](https://platform.deepseek.com/docs)
- [Agent Tools Documentation](/README.md)
- [Tool Development Guide](tool_development.md) 