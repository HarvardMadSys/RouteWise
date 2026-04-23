# Code Examples

Essential examples for using the HybridInference API.

## Installation

**Python:**
```bash
pip install openai
```

**JavaScript/TypeScript:**
```bash
npm install openai
```

## Basic Chat Completion

### Python

```python
import openai

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning in simple terms"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### JavaScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://freeinference.org/v1',
  apiKey: 'your-api-key-here',
});

const response = await client.chat.completions.create({
  model: 'llama-3.3-70b-instruct',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain machine learning in simple terms' }
  ],
  temperature: 0.7,
  max_tokens: 1000
});

console.log(response.choices[0].message.content);
```

### curl

```bash
curl https://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain machine learning in simple terms"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

---

## Streaming Responses

Stream responses in real-time for better user experience.

### Python

```python
import openai

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

stream = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[
        {"role": "user", "content": "Write a short story about a robot"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### JavaScript

```javascript
const stream = await client.chat.completions.create({
  model: 'llama-3.3-70b-instruct',
  messages: [
    { role: 'user', content: 'Write a short story about a robot' }
  ],
  stream: true
});

for await (const chunk of stream) {
  if (chunk.choices[0]?.delta?.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}
```

### curl

```bash
cURL https://freeinference.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Write a short story about a robot"}],
    "stream": true
  }' \
  --no-buffer
```

---

## Function Calling

Enable the model to call functions you define.

### Python

```python
import openai
import json

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    print(f"Function to call: {function_name}")
    print(f"Arguments: {function_args}")

    # Here you would call your actual function
    # result = get_weather(**function_args)
```

### JavaScript

```javascript
const tools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather in a location',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'City name, e.g. San Francisco'
          },
          unit: {
            type: 'string',
            enum: ['celsius', 'fahrenheit']
          }
        },
        required: ['location']
      }
    }
  }
];

const response = await client.chat.completions.create({
  model: 'llama-3.3-70b-instruct',
  messages: [
    { role: 'user', content: "What's the weather in Paris?" }
  ],
  tools: tools,
  tool_choice: 'auto'
});

if (response.choices[0].message.tool_calls) {
  const toolCall = response.choices[0].message.tool_calls[0];
  const functionName = toolCall.function.name;
  const functionArgs = JSON.parse(toolCall.function.arguments);

  console.log(`Function to call: ${functionName}`);
  console.log(`Arguments:`, functionArgs);

  // Here you would call your actual function
  // const result = await getWeather(functionArgs);
}
```

---

## Structured Output (JSON Mode)

Force the model to respond with valid JSON.

### Python

```python
import openai
import json

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Extract the name, age, and occupation from: John is a 30-year-old software engineer. Return as JSON."
        }
    ],
    response_format={"type": "json_object"}
)

# Parse the JSON response
content = response.choices[0].message.content

# Note: Response may be wrapped in markdown code blocks
# Handle both pure JSON and markdown-wrapped JSON
try:
    result = json.loads(content)
except json.JSONDecodeError:
    # Extract from markdown if needed
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        content = content[start:end].strip()
        result = json.loads(content)

print(result)
# Output: {'name': 'John', 'age': 30, 'occupation': 'software engineer'}
```

### JavaScript

```javascript
const response = await client.chat.completions.create({
  model: 'llama-3.3-70b-instruct',
  messages: [
    {
      role: 'user',
      content: 'Extract the name, age, and occupation from: John is a 30-year-old software engineer. Return as JSON.'
    }
  ],
  response_format: { type: 'json_object' }
});

let content = response.choices[0].message.content;

// Handle markdown-wrapped JSON
try {
  const result = JSON.parse(content);
  console.log(result);
} catch (error) {
  if (content.includes('```json')) {
    const start = content.indexOf('```json') + 7;
    const end = content.indexOf('```', start);
    content = content.substring(start, end).trim();
    const result = JSON.parse(content);
    console.log(result);
  }
}
```

---

## Tips

### Temperature Settings

- **0.0 - 0.3**: Deterministic, focused (good for factual tasks, code generation)
- **0.7**: Balanced (general use)
- **0.9 - 1.5**: Creative, diverse (good for storytelling, brainstorming)

### Max Tokens

Always set `max_tokens` to control response length and costs:

```python
response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Summarize this article..."}],
    max_tokens=500  # Limit response length
)
```

### Choosing Models

- **llama-3.3-70b-instruct**: Best for general tasks, long context
- **llama-4-scout**: Fastest inference
- **gemini-2.5-flash**: Multimodal, high throughput
- **glm-4.5**: Chinese language support

See the [Models](models.md) page for full details.
