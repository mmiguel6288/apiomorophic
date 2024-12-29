# test_apiomorphic.py
import pytest
import base64
import json
from typing import Dict, Any
from apiomorphic import translate, FromOpenAi, FromAnthropic, format_tool_schema

# Fixtures
@pytest.fixture
def sample_base64_image():
    return base64.b64encode(b"fake_image_data").decode('utf-8')

@pytest.fixture
def basic_messages():
    return {
        "openai": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "max_tokens": 100
        },
        "anthropic": {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "system": "You are a helpful assistant.",
            "max_tokens": 100
        }
    }

# Basic Translation Tests
def test_translate_function():
    assert translate('openai', 'anthropic') == FromOpenAi.ToAnthropic
    assert translate('anthropic', 'openai') == FromAnthropic.ToOpenAi
    with pytest.raises(Exception):
        translate('invalid', 'format')

# Message Conversion Tests
def test_basic_message_conversion_openai_to_anthropic(basic_messages):
    converter = translate('openai', 'anthropic')
    result = converter.convert(basic_messages['openai'])
    assert result == basic_messages['anthropic']

def test_tool_conversion_openai_to_anthropic():
    openai_tool_msg = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}'
                    }
                }]
            }
        ]
    }
    
    expected_anthropic = {
        "messages": [
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "London"}
                }]
            }
        ]
    }
    
    converter = translate('openai', 'anthropic')
    result = converter.convert(openai_tool_msg)
    assert result == expected_anthropic

def test_image_conversion_anthropic_to_openai(sample_base64_image):
    anthropic_msg = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": sample_base64_image
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
            }
        ]
    }
    
    expected_openai = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{sample_base64_image}",
                    "detail": "auto"
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
            }
        ]
    }
    
    converter = translate('anthropic', 'openai')
    result = converter.convert_message(anthropic_msg)
    assert result[0] == expected_openai

# Tool Schema Tests
def test_tool_schema_conversion():
    openai_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
    
    expected_anthropic = {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
    
    converter = FromOpenAi.ToAnthropic
    result = converter.convert_tool_schema(openai_schema)
    assert result == expected_anthropic

def test_empty_tool_schema():
    openai_schema = {
        "type": "function",
        "function": {
            "name": "empty_tool"
        }
    }
    
    expected_anthropic = {
        "name": "empty_tool",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
    
    converter = FromOpenAi.ToAnthropic
    result = converter.convert_tool_schema(openai_schema)
    assert result == expected_anthropic

def test_tool_response_conversion_openai_to_anthropic():
    openai_response = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Sunny, 22°C"
    }
    
    expected_anthropic = {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": "Sunny, 22°C"
        }]
    }
    
    converter = translate('openai', 'anthropic')
    result = converter.convert_message(openai_response)[0]
    assert result == expected_anthropic

# Complex Message Tests
def test_nested_tool_calls():
    nested_msg = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "func1",
                            "arguments": json.dumps({"arg1": "val1"})
                        }
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "func2",
                            "arguments": json.dumps({"arg2": "val2"})
                        }
                    }
                ]
            }
        ]
    }
    converter = translate('openai', 'anthropic')
    result = converter.convert(nested_msg)
    assert len(result["messages"]) == 2
    assert all(msg["role"] == "assistant" for msg in result["messages"])

# Error Cases
def test_invalid_message_format():
    invalid_msg = {"role": "invalid", "content": "test"}
    with pytest.raises(Exception):
        converter = translate('invalid', 'anthropic')
        converter.convert_message(invalid_msg)

# Integration Tests
@pytest.mark.integration
def test_full_conversation_flow():
    conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London"}'
                    }
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "weather_1",
                "content": "Sunny, 22°C"
            },
            {"role": "assistant", "content": "The weather in London is sunny and 22°C."}
        ]
    }
    
    # Test both conversion directions
    openai_to_anthropic = translate('openai', 'anthropic').convert(conversation)
    anthropic_to_openai = translate('anthropic', 'openai').convert(openai_to_anthropic)
    
    # Verify system message handling
    assert "system" in openai_to_anthropic
    assert openai_to_anthropic["system"] == "You are a helpful assistant."
    
    # Verify tool calls conversion
    assert any("tool_use" in str(msg.get("content")) for msg in openai_to_anthropic["messages"])
    assert any("tool_calls" in str(msg) for msg in anthropic_to_openai["messages"])

def test_format_tool_schema():
    # Example tool data
    tools = [
        (
            "math.add",
            "Add two numbers together",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        ),
        (
            "text.uppercase",
            "Convert text to uppercase",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        )
    ]

    # Test OpenAI format
    openai_result = format_tool_schema("openai", tools, strict=True)
    assert openai_result == [
        {
            "type": "function",
            "function": {
                "name": "math.add",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                },
                "strict": True
            }
        },
        {
            "type": "function",
            "function": {
                "name": "text.uppercase",
                "description": "Convert text to uppercase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"]
                },
                "strict": True
            }
        }
    ]

    # Test Anthropic format
    anthropic_result = format_tool_schema("anthropic", tools)
    assert anthropic_result == [
        {
            "name": "math.add",
            "description": "Add two numbers together",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        },
        {
            "name": "text.uppercase",
            "description": "Convert text to uppercase",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    ]

    # Test invalid format
    try:
        format_tool_schema("invalid", tools)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
