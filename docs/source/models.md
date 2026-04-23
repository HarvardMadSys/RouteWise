# Available Models

HybridInference provides access to multiple state-of-the-art LLM models.

## Model Overview

| Model ID | Name | Context Length | Pricing |
|----------|------|----------------|---------|
| `llama-3.3-70b-instruct` | Llama 3.3 70B Instruct | 131K tokens | Free |
| `llama-4-scout` | Llama 4 Scout | 128K tokens | Free |
| `llama-4-maverick` | Llama 4 Maverick | 128K tokens | Free |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1M tokens | Free |
| `gemini-2.5-flash-preview-09-2025` | Gemini 2.5 Flash Preview | 1M tokens | Free |
| `glm-4.5` | GLM-4.5 | 128K tokens | Free |
| `gpt-5` | GPT-5 | 128K tokens | Free |
| `custom-model-alpha` | Claude Opus 4.1 | 200K tokens | Free |
| `custom-model-beta` | GPT-5 (Azure) | 400K tokens | Free |

## Model Details

### Llama 3.3 70B Instruct

**Model ID:** `llama-3.3-70b-instruct`

High-performance open-source model optimized for instruction following.

**Key Features:**
- Context length: 131,072 tokens
- Max output: 8,192 tokens
- Function calling support
- Structured output (JSON mode)
- Quantization: bf16

**Best For:**
- General purpose chat
- Long-form content generation
- Code generation
- Instruction following

**Example:**
```python
response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=2048
)
```

---

### Llama 4 Scout

**Model ID:** `llama-4-scout`

Efficient MoE (Mixture of Experts) model for fast inference.

**Key Features:**
- Context length: 128,000 tokens
- Max output: 16,384 tokens
- Function calling support
- Structured output
- Quantization: fp8

**Best For:**
- Fast inference scenarios
- Cost-effective deployments
- Production workloads

---

### Llama 4 Maverick

**Model ID:** `llama-4-maverick`

Advanced MoE (Mixture of Experts) model for complex tasks.

**Key Features:**
- Context length: 128,000 tokens
- Max output: 16,384 tokens
- Function calling support
- Structured output
- Quantization: fp8

**Best For:**
- Complex reasoning tasks
- Long-form generation
- Production workloads with high quality requirements

---

### Gemini 2.5 Flash

**Model ID:** `gemini-2.5-flash`

Google's fast and efficient model.

**Key Features:**
- Fast inference speed
- High throughput
- Large context window
- Production-ready

**Best For:**
- Real-time applications
- High-volume workloads
- Quick responses needed

---

### GLM-4.5

**Model ID:** `glm-4.5`

Bilingual model optimized for Chinese and English.

**Best For:**
- Chinese language tasks
- Bilingual applications
- Cross-language translation

---

### GPT-5

**Model ID:** `gpt-5`

Latest OpenAI flagship model.

**Best For:**
- Complex reasoning
- Advanced code generation
- Research applications

---

### Claude Opus 4.1

**Model ID:** `custom-model-alpha`

Anthropic's most capable model for complex tasks.

**Best For:**
- Long-form writing
- Advanced analysis
- Research and development

---

## Using Different Models

Simply change the `model` parameter in your request:

```python
import openai

client = openai.OpenAI(
    base_url="https://freeinference.org/v1",
    api_key="your-api-key-here"
)

# Use Llama 3.3
response = client.chat.completions.create(
    model="llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Switch to Gemini
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Model Selection Guide

**For general chat and instructions:**
- `llama-3.3-70b-instruct` - Best balance of quality and speed
- `llama-4-maverick` - High quality, complex tasks

**For fast inference:**
- `llama-4-scout` - Optimized for speed
- `gemini-2.5-flash` - High throughput, real-time use

**For Chinese language:**
- `glm-4.5` - Bilingual Chinese/English support

**For advanced reasoning:**
- `gpt-5` - Latest OpenAI capabilities
- `custom-model-alpha` (Claude Opus 4.1) - Complex analysis and long-form writing
