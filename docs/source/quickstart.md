# Quick Start

Get started with HybridInference API in 5 minutes.

## Overview

HybridInference provides an OpenRouter-compatible API for accessing multiple LLM models through a single endpoint.

**API Endpoint:** `https://freeinference.org/v1`

## Get Your API Key

Contact the team to get your API key, or use the example key for testing:

```bash
export HYBRIDINFERENCE_API_KEY="your-api-key-here"
```

## Make Your First Request

### Using curl

```bash
curl http://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HYBRIDINFERENCE_API_KEY" \
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

### Using Python

```python
import openai

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
```

### Using JavaScript/TypeScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://freeinference.org/v1',
  apiKey: 'your-api-key-here',
});

async function main() {
  const response = await client.chat.completions.create({
    model: 'llama-3.3-70b-instruct',
    messages: [
      { role: 'user', content: 'What is the capital of France?' }
    ],
  });

  console.log(response.choices[0].message.content);
}

main();
```

## Streaming Responses

```python
import openai

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

stream = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Check Available Models

```bash
curl http://freeinference.org/v1/models \
  -H "Authorization: Bearer $HYBRIDINFERENCE_API_KEY"
```

## Next Steps

- [View all available models](models.md)
- [API reference](api-reference.md)
- [Code examples](examples.md)
