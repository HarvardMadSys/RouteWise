HybridInference API Documentation
===================================

**OpenRouter-compatible API for accessing multiple LLM models**

Get started with HybridInference in minutes. Our API provides seamless access to
state-of-the-art language models including Llama 3.3, Llama 4, Gemini, GPT-5, and Claude.

Quick Links
-----------

* :doc:`quickstart` - Get started in 5 minutes
* :doc:`models` - View available models
* :doc:`examples` - Code examples in Python, JavaScript, and more
* :doc:`integrations` - Configure with Cursor, Windsurf, and other coding agents
* :doc:`api-reference` - Complete API reference

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   :hidden:

   quickstart
   models
   examples
   integrations
   api-reference

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:
   :hidden:

   developer/installation
   developer/deployment
   developer/architecture
   developer/routing
   developer/adding-models
   developer/configuration
   developer/database
   developer/openrouter
   developer/freeinference
   developer/fasrc
   developer/contributing

Key Features
------------

**Fast & Reliable**
   Low-latency inference with automatic failover

**OpenRouter Compatible**
   Drop-in replacement for OpenRouter API

**Multiple Models**
   Access Llama, Gemini, GPT, and Claude models

**Production Ready**
   Built for scale with monitoring and observability

Getting Started
---------------

1. **Get your API key** (contact the team)

2. **Install the OpenAI SDK:**

   .. code-block:: bash

      pip install openai

3. **Make your first request:**

   .. code-block:: python

      import openai

      client = openai.OpenAI(
          base_url="https://freeinference.org/v1",
          api_key="your-api-key-here"
      )

      response = client.chat.completions.create(
          model="llama-3.3-70b-instruct",
          messages=[{"role": "user", "content": "Hello!"}]
      )

      print(response.choices[0].message.content)

See the :doc:`quickstart` guide for more details.

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Model
     - Context Length
     - Pricing
   * - Llama 3.3 70B Instruct
     - 131K tokens
     - Free
   * - Llama 4 Maverick
     - 128K tokens
     - Free
   * - Gemini 2.5 Flash
     - 1M tokens
     - Free
   * - GPT-5
     - 128K tokens
     - Free

See the complete :doc:`models` list for all available models.

Support
-------

Need help? Check out:

* :doc:`examples` - Code examples
* :doc:`api-reference` - API documentation
* GitHub Issues - Report bugs or request features
