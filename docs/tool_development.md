# Tool Development Guide

This guide explains how to create new tools that work with both Claude and Deepseek Reasoner.

## Tool Structure

Every tool must inherit from the base `Tool` class and implement the required interface:

```python
from typing import Dict, Any
from tools.tool_base import Tool

class YourTool(Tool):
    name = "your_tool"  # Unique identifier for your tool
    description = "A clear description of what your tool does"
    
    # Define the expected input input_schema
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform",
                "enum": ["action1", "action2"]
            },
            "value": {
                "type": "number",
                "description": "The value to use for the action"
            }
        },
        "required": ["action"]
    }

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given input_schema."""
        try:
            # Your implementation here
            result = self._perform_action(params)
            
            return {
                "type": "tool_result",
                "content": {
                    "status": "success",
                    "message": "Action completed successfully",
                    "data": result
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

## Required Components

### 1. Tool Properties

- `name`: Unique identifier for your tool
- `description`: Clear, detailed description of what your tool does
- `input_schema`: JSONSchema defining expected input_schema

### 2. Input Schema

Your schema must define:
- Parameter names and types
- Clear descriptions for each parameter
- Required vs optional input_schema
- Any constraints (enums, ranges, etc.)

Example schema:
```python
input_schema = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "The action to perform",
            "enum": ["create", "read", "update", "delete"]
        },
        "id": {
            "type": "string",
            "description": "Resource identifier"
        },
        "data": {
            "type": "object",
            "description": "Data for create/update operations",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"}
            }
        }
    },
    "required": ["action", "id"]
}
```

### 3. Run Method

The `run` method must:
- Accept a dictionary of input_schema
- Validate inputs against schema
- Return results in the standard format
- Handle errors gracefully

## Response Format

All tools must return responses in this format:

```python
{
    "type": "tool_result",
    "content": {
        "status": "success" | "error",
        "message": str,  # Human-readable description
        "data": Any     # Optional: Additional data
    }
}
```

## Best Practices

1. **Input Validation**
```python
def run(self, params):
    # Validate required input_schema
    if "action" not in params:
        return {
            "type": "tool_result",
            "content": {
                "status": "error",
                "message": "Missing required parameter: action"
            }
        }
```

2. **Error Handling**
```python
def run(self, params):
    try:
        result = self._perform_action(params)
    except ValueError as e:
        return {
            "type": "tool_result",
            "content": {
                "status": "error",
                "message": f"Invalid input: {str(e)}"
            }
        }
    except Exception as e:
        return {
            "type": "tool_result",
            "content": {
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }
        }
```

3. **Clear Documentation**
```python
class YourTool(Tool):
    """
    A tool for performing specific actions.
    
    Capabilities:
    - Action 1: Description of what it does
    - Action 2: Description of what it does
    
    Example:
        tool = YourTool()
        result = tool.run({
            "action": "action1",
            "value": 42
        })
    """
```

## Testing Your Tool

1. Create a test file in the tests directory:
```python
# tests/test_your_tool.py
import pytest
from your_tool import YourTool

def test_basic_functionality():
    tool = YourTool()
    result = tool.run({"action": "action1", "value": 42})
    assert result["type"] == "tool_result"
    assert result["content"]["status"] == "success"

def test_error_handling():
    tool = YourTool()
    result = tool.run({"action": "invalid"})
    assert result["content"]["status"] == "error"
```

2. Run the tests:
```bash
# Run specific tool tests
pytest tests/test_your_tool.py

# Run with LLM integration
pytest -m llm tests/test_your_tool.py
```

## Integration Testing

Test your tool with both Claude and Deepseek:

```python
def test_llm_integration(your_tool):
    # Test with Deepseek
    wrapper = DeepseekToolWrapper()
    wrapper.register_tool(your_tool)
    result = wrapper.execute("Use the tool to perform action1")
    assert "success" in result

    # Test with Claude
    # Add Claude-specific testing
```

## Common Pitfalls

1. **Unclear Descriptions**
   - Bad: "Performs an action"
   - Good: "Creates a new resource with the specified name and value"

2. **Missing Parameter Descriptions**
   - Bad: `"value": {"type": "number"}`
   - Good: `"value": {"type": "number", "description": "The value to use (between 0 and 100)"}`

3. **Inconsistent Error Handling**
   - Always use the standard error format
   - Provide actionable error messages

## Security Considerations

1. **Input Sanitization**
```python
def _sanitize_input(self, value: str) -> str:
    """Clean user input to prevent injection attacks."""
    # Your sanitization logic
    return cleaned_value
```

2. **Permission Checking**
```python
def _check_permissions(self, action: str) -> bool:
    """Verify the tool has permission for the action."""
    # Your permission logic
    return is_allowed
```

## Examples

See these example tools:
- [Computer Tool](/tools/computer_tool.py): System interaction example for controlling mouse, keyboard, and other computer functions 