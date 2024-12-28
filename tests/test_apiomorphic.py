# test_apiomorphic.py
import pytest
import base64
from apiomorphic import translate, FromOpenAi, FromAnthropic

# Test data
SAMPLE_BASE64_IMAGE = base64.b64encode(b"fake_image_data").decode('utf-8')

def test_translate_function():
    assert translate('openai', 'anthropic') == FromOpenAi.ToAnthropic
    assert translate('anthropic', 'openai') == FromAnthropic.ToOpenAi
    with pytest.raises(Exception):
        translate('invalid', 'format')

def test_basic_message_conversion_openai_to_anthropic():
    openai_messages = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "max_tokens": 100
    }
    
    expected_anthropic = {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "system": "You are a helpful assistant.",
        "max_tokens": 100
    }
    
    converter = translate('openai', 'anthropic')
    result = converter.convert(openai_messages)
    assert result == expected_anthropic

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

def test_image_conversion_anthropic_to_openai():
    anthropic_msg = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": SAMPLE_BASE64_IMAGE
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
                    "url": f"data:image/jpeg;base64,{SAMPLE_BASE64_IMAGE}",
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
