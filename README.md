# apiomorphic

apiomorphic is a Python library for seamlessly converting between different AI API formats, currently supporting OpenAI and Anthropic APIs.

## Installation

```
pip install git+https://github.com/yourusername/apiomorphic.git
```

Or clone and install locally:
```
git clone https://github.com/yourusername/apiomorphic.git
cd apiomorphic
pip install -e .
```

## Features

- Convert between OpenAI and Anthropic API formats
- Support for:
  - Text messages
  - Tool calls/function calling
  - Image handling
  - System messages
- Maintains message order and context
- Preserves all relevant metadata

## Usage

### Basic Usage

```python
from apiomorphic import translate

# Convert from OpenAI to Anthropic format
converter = translate('openai', 'anthropic')
anthropic_params = converter.convert(openai_params)

# Convert from Anthropic to OpenAI format
converter = translate('anthropic', 'openai')
openai_params = converter.convert(anthropic_params)
```

### Advanced Usage

```python
from apiomorphic import FromOpenAi, FromAnthropic

# Convert specific message
anthropic_message = FromOpenAi.ToAnthropic.convert_message(openai_message)

# Convert tool schema
anthropic_schema = FromOpenAi.ToAnthropic.convert_tool_schema(openai_tool_schema)
```

## API Reference

### translate(source: str, target: str)

Returns a converter class for translating between specified API formats.

### FromOpenAi.ToAnthropic

- `convert(api_params)`: Convert complete API parameters
- `convert_message(msg)`: Convert single message
- `convert_tool_schema(tool_schema_entry)`: Convert tool/function schema
- `convert_vision(msg)`: Convert vision-related content

### FromAnthropic.ToOpenAi

- `convert(api_params)`: Convert complete API parameters
- `convert_message(msg)`: Convert single message
- `convert_tool_schema(tool_schema_entry, strict=False)`: Convert tool/function schema

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
