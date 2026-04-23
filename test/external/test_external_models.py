"""Async smoke tester for OpenRouter-compatible model endpoints."""

import asyncio
import json
import time
from typing import Any

import httpx


class ModelTester:
    def __init__(self, base_url: str = "http://freeinference.org"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_models(self) -> list[dict[str, Any]]:
        """Fetch available models from the API"""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"❌ Failed to fetch models: {e}")
            return []

    async def test_model(
        self, model_id: str, test_message: str = "Hello! Can you introduce yourself briefly?"
    ) -> dict[str, Any]:
        """Test a single model with a simple message"""
        start_time = time.time()

        # Use higher max_tokens for thinking models (e.g., Gemini preview)
        # which may consume tokens for internal reasoning
        max_tokens = 1000 if "preview" in model_id else 200

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": test_message}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
                )
                usage = data.get("usage", {})

                return {
                    "status": "success",
                    "model": model_id,
                    "response": content[:200] + "..." if len(content) > 200 else content,
                    "usage": usage,
                    "latency_ms": int(elapsed * 1000),
                    "status_code": response.status_code,
                }
            else:
                error_data = (
                    response.json()
                    if response.headers.get("content-type", "").startswith("application/json")
                    else response.text
                )
                return {
                    "status": "error",
                    "model": model_id,
                    "error": error_data,
                    "latency_ms": int(elapsed * 1000),
                    "status_code": response.status_code,
                }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "status": "failed",
                "model": model_id,
                "error": str(e),
                "latency_ms": int(elapsed * 1000),
                "status_code": None,
            }

    async def test_all_models(self, custom_message: str | None = None) -> list[dict[str, Any]]:
        """Test all available models"""
        models = await self.get_models()
        if not models:
            return []

        print(f"🚀 Testing {len(models)} models...\n")

        # Test all models concurrently
        tasks = []
        for model in models:
            model_id = model["id"]
            message = (
                custom_message
                or f"Hello! I'm testing the {model['name']} model. Can you respond briefly?"
            )
            tasks.append(self.test_model(model_id, message))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "status": "failed",
                        "model": models[i]["id"],
                        "error": str(result),
                        "latency_ms": 0,
                        "status_code": None,
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def print_results(self, results: list[dict[str, Any]]):
        """Print test results in a human-readable format"""
        print("=" * 80)
        print("🧪 MODEL INFERENCE TEST RESULTS")
        print("=" * 80)

        successful = 0
        failed = 0

        for result in results:
            model = result["model"]
            status = result["status"]
            latency = result["latency_ms"]

            if status == "success":
                successful += 1
                print(f"✅ {model}")
                print(f"   Latency: {latency}ms")
                if result.get("usage"):
                    usage = result["usage"]
                    print(
                        f"   Tokens: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion"
                    )
                print(f"   Response: {result['response']}")
                print()

            else:
                failed += 1
                print(f"❌ {model}")
                print(f"   Status: {status}")
                print(f"   Latency: {latency}ms")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                if result.get("status_code"):
                    print(f"   HTTP Status: {result['status_code']}")
                print()

        print("=" * 80)
        print(f"📊 SUMMARY: {successful} successful, {failed} failed")
        print("=" * 80)

    async def close(self):
        """Clean up the HTTP client"""
        await self.client.aclose()


async def main():
    """Main function to run the model tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Test model inference on freeinference.org")
    parser.add_argument("--url", default="http://freeinference.org", help="Base URL for the API")
    parser.add_argument("--message", help="Custom test message (optional)")
    parser.add_argument("--model", help="Test a specific model only")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    tester = ModelTester(args.url)

    try:
        if args.model:
            # Test single model
            result = await tester.test_model(args.model, args.message)
            if args.json:
                print(json.dumps([result], indent=2))
            else:
                tester.print_results([result])
        else:
            # Test all models
            results = await tester.test_all_models(args.message)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                tester.print_results(results)
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
