# IDE & Coding Agent Integrations

Learn how to configure Free Inference with popular coding agents and IDEs.

---

## Codex

[Codex](https://codex.so/) is a powerful AI coding assistant.

### Configuration Steps

1. Create or edit the Codex configuration file at `~/.codex/config.toml`

2. Add the following configuration:

```toml
# ~/.codex/config.toml

model = "glm-4.6"

model_provider = "hybrid_inference"
model_reasoning_effort = "high"

[model_providers.hybrid_inference]
name = "HybridInference"
base_url = "https://freeinference.org/v1"
wire_api = "chat"
env_http_headers = { "X-Session-ID" = "CODEX_SESSION_ID", "Authorization" = "HYBRID_INFERENCE_API_KEY" }
request_max_retries = 5
```

3. Set up environment variables in your shell configuration file (`~/.zshrc` or `~/.bashrc`):

```bash
# Add these lines to ~/.zshrc or ~/.bashrc

# Generate unique session ID for each shell session
export CODEX_SESSION_ID="$(date +%Y%m%d-%H%M%S)-$(uuidgen)"

# Your Free Inference API key (note: include "Bearer " prefix)
export HYBRID_INFERENCE_API_KEY="Bearer your-api-key-here"
```

4. Reload your shell configuration:

```bash
# For zsh
source ~/.zshrc

# For bash
source ~/.bashrc
```

5. Start using Codex with Free Inference!


## Cursor

[Cursor](https://cursor.sh/) is an AI-powered code editor built on VS Code.

### Configuration Steps

1. Open Cursor Settings
   - **macOS**: `Cmd + ,`
   - **Windows/Linux**: `Ctrl + ,`

2. Navigate to **API Keys** section in the settings

3. Find the **OpenAI API Key** field and enter your Free Inference API key

4. Click on **Override OpenAI Base URL**

5. Enter the base URL:
   ```
   https://freeinference.org/v1
   ```

6. Enable the **OpenAI API Key** toggle button

7. (Optional) Select your preferred model:
   ```
   Model: glm-4.6
   ```

8. Save and start using Free Inference models in Cursor!

---

## Roo Code & Kilo Code

[Roo Code](https://roo.dev/) and [Kilo Code](https://kilo.dev/) are AI coding assistants with similar configuration.

### Configuration Steps

1. Install the Roo Code or Kilo Code extension/plugin in your IDE (VS Code or JetBrains)

2. Open the settings (click the settings icon in the extension panel)

3. In **API Provider**, select **OpenAI Compatible**

4. Configure the connection:
   ```
   Base URL: https://freeinference.org/v1
   API Key: your-api-key-here
   ```

5. Select your preferred model (e.g., `glm-4.6`, `llama-3.3-70b-instruct`, etc.)

6. Save settings and start using with Free Inference!

---

## Troubleshooting

### Connection Issues

If you encounter connection errors:

1. Verify your API key is correct
2. Check the base URL is exactly: `https://freeinference.org/v1`
3. Ensure your firewall allows HTTPS connections
4. Try testing with curl first:

```bash
curl https://freeinference.org/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Model Not Found

If you get "model not found" errors:

- Check the [available models](models.md) list
- Ensure the model name is exactly as listed (case-sensitive)
- Some models may have usage limits

### Codex-Specific Issues

**Environment variables not loaded:**
- Make sure you've reloaded your shell configuration after editing `~/.zshrc` or `~/.bashrc`
- Verify variables are set: `echo $HYBRID_INFERENCE_API_KEY`

**Session ID issues:**
- The session ID is auto-generated each time you start a new shell
- If needed, you can manually set it: `export CODEX_SESSION_ID="custom-session-id"`

---

## Need Help?

- Check our [API Reference](api-reference.md)
- View [Code Examples](examples.md)
- Report issues on GitHub
