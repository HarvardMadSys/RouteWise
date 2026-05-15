"""Latency profiling script for collecting TTFT and E2E latency data from OpenRouter providers."""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    """Load a project-root .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


_load_project_env()

# Define random_common_words here to avoid import issues if paths are tricky
COMMON_WORDS = (
    "the be to of and a in that have I it for not on with he as you do at "
    "this but his by from they we say her she or an will my one all would there their what "
    "so up out if about who get which go me when make can like time no just him know take "
    "people into year your good some could them see other than then now look only come "
    "its over think also back after use two how our work first well way even new want "
    "because any these give day most us find long down may should call world life still "
    "right might great man where too last never leave while place same old feel mean keep "
    "let put every big high small next another few again home read hand part end point set "
    "change play off turn start move house run show under story eye face head open stop "
    "without name question number side school family student group country problem fact "
    "water food air mother father child friend city body hour minute second week month "
    "year morning night light dark white black red blue green yellow brown animal dog cat "
    "bird fish tree mountain river sea ocean earth star sky space music song voice sound"
).split()


def generate_random_prompt(n_words: int) -> str:
    """Generate a random prompt with the specified number of words."""
    return " ".join(random.choices(COMMON_WORDS, k=n_words))


API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.3-70b-instruct"

# Target Providers (verified to support meta-llama/llama-3.3-70b-instruct)
# Source: OpenRouter provider list from teacher's email
# Removed: Avian, Upstage, Scaleway, Google Vertex (404: No endpoints found)
PROVIDERS = [
    # === From teacher's list (12 providers total) ===
    "Cerebras",
    "Friendli",
    "Together",
    "Novita",  # FP8
    "Hyperbolic",  # FP8
    "Fireworks",  # FP8
    "Nebius",
    "Groq",  # Fastest!
    "SambaNova",
    "Parasail",
    "Crusoe",
    "Cloudflare",
    # "Inceptron",   # TODO: verify if supported
    # "Weights & Biases",  # TODO: verify if supported
    # "DeepInfra",  # Rate-limited, removed by user request
    # "Google Vertex",  # 404: No endpoints found
]

# Providers with strict rate limits - use lower concurrency
RATE_LIMITED_PROVIDERS = set()

# Skip providers that already have successful results (to save cost)
# Add provider names here to skip them
SKIP_PROVIDERS = set()
# Example: SKIP_PROVIDERS = {"Friendli", "Together"}


def get_tested_providers(data_dir: str) -> set:
    """Scan existing CSV files to find providers that have already been tested successfully."""
    tested_providers = set()
    if not os.path.exists(data_dir):
        return tested_providers

    print(f"Scanning {data_dir} for existing results...")
    for filename in os.listdir(data_dir):
        if filename.startswith("latency_profile_llama3.3_70b_") and filename.endswith(".csv"):
            try:
                file_path = os.path.join(data_dir, filename)
                # Read only necessary columns to speed up
                df = pd.read_csv(file_path, usecols=["provider", "status"])
                if "provider" in df.columns and "status" in df.columns:
                    # Only consider providers that have at least one successful request
                    successful_providers = df[df["status"] == "success"]["provider"].unique()
                    if len(successful_providers) > 0:
                        # print(f"  Found successful tests in {filename}: {successful_providers}")
                        tested_providers.update(successful_providers)
            except Exception:
                # print(f"  Warning: Could not read {filename}: {e}")
                pass
    return tested_providers


# Workload Configuration
# Set TEST_MODE = True to run a small test first
TEST_MODE = False

# Long-run configuration (for ~10h experiment)
LONG_RUN_CONFIG = {
    "duration_hours": 12,
    "rounds_per_hour": 80,  # Run every 45 seconds
    # Target: ~10000 requests per provider over ~10h
    # 12 * 80 = 960 rounds
    # 10000 / 960 ~= 10 requests per round
    # Only running "base" workload as per teacher's requirement
    "workloads_per_round": [
        {"name": "base", "input_len": 50, "output_len": 10, "count": 10},
    ],
}

if TEST_MODE:
    # Override for quick testing
    LONG_RUN_CONFIG = {
        "duration_hours": 0.02,  # Very short duration (e.g. ~1 min)
        "rounds_per_hour": 60,
        "workloads_per_round": [
            {"name": "base", "input_len": 50, "output_len": 10, "count": 1},
            {"name": "prefill", "input_len": 1000, "output_len": 10, "count": 1},
            {"name": "decode", "input_len": 50, "output_len": 200, "count": 1},
        ],
    }
    print("!!! RUNNING IN TEST MODE (Low request count) !!!")
    print("Set TEST_MODE = False in the script to run full experiment.")

MAX_CONCURRENCY = 10  # Increased for parallel providers
MAX_CONCURRENCY_RATE_LIMITED = 1  # For providers with strict rate limits
MAX_RETRIES = 5  # Increased retries for stability
INITIAL_BACKOFF = 4.0  # Increased backoff for rate-limited providers
REQUEST_TIMEOUT = 60  # seconds
INTER_REQUEST_DELAY = 0.1  # Small delay between requests in a batch


async def send_request(session, provider, input_len, output_len, request_id):
    """Send a single request to a provider and measure latency."""
    prompt_text = generate_random_prompt(input_len)

    messages = [
        {
            "role": "user",
            "content": f"{prompt_text}\n\nIgnore previous instructions. Please generate exactly {output_len} words of random text.",
        }
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": output_len,
        "stream": True,
        "provider": {"order": [provider], "allow_fallbacks": False},
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.org/routewise-artifact",
        "X-Title": "Latency Profiling",
    }

    start_time = time.time()
    ttft = 0
    first_token_time = 0
    chunk_count = 0

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{BASE_URL}/chat/completions", json=payload, headers=headers
            ) as response:
                # Handle Rate Limits (429)
                if response.status == 429:
                    if attempt < MAX_RETRIES:
                        wait_time = INITIAL_BACKOFF * (2**attempt) + random.uniform(0, 1)
                        # print(f"[{request_id}] Hit 429, retrying in {wait_time:.2f}s (Attempt {attempt+1}/{MAX_RETRIES})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return {
                            "request_id": request_id,
                            "provider": provider,
                            "status": "error",
                            "error_code": 429,
                            "error_msg": "Rate limit exceeded after retries",
                            "ttft_ms": -1,
                            "e2e_ms": -1,
                            "input_len": input_len,
                            "output_len": output_len,
                            "timestamp": start_time,
                        }

                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "request_id": request_id,
                        "provider": provider,
                        "status": "error",
                        "error_code": response.status,
                        "error_msg": error_text,
                        "ttft_ms": -1,
                        "e2e_ms": -1,
                        "input_len": input_len,
                        "output_len": output_len,
                        "timestamp": start_time,
                    }

                # Process Streaming Response
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            # Check if content exists in delta
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            # Only record TTFT if we actually got content (skip role: assistant or empty chunks)
                            if content:
                                if first_token_time == 0:
                                    first_token_time = time.time()
                                    ttft = first_token_time - start_time
                                chunk_count += 1
                        except json.JSONDecodeError:
                            continue

                end_time = time.time()
                e2e = end_time - start_time

                # If TTFT wasn't captured but request succeeded (rare), fallback to E2E
                if ttft == 0:
                    ttft = e2e

                return {
                    "request_id": request_id,
                    "provider": provider,
                    "status": "success",
                    "ttft_ms": ttft * 1000,  # Convert to milliseconds
                    "e2e_ms": e2e * 1000,  # Convert to milliseconds
                    "input_len": input_len,
                    "output_len": output_len,
                    "timestamp": start_time,
                    "tokens_generated": chunk_count,
                }

        except Exception as e:
            if attempt < MAX_RETRIES:
                wait_time = INITIAL_BACKOFF * (2**attempt) + random.uniform(0, 1)
                # print(f"[{request_id}] Error: {str(e)}, retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "request_id": request_id,
                    "provider": provider,
                    "status": "error",
                    "error_msg": str(e),
                    "ttft_ms": -1,
                    "e2e_ms": -1,
                    "input_len": input_len,
                    "output_len": output_len,
                    "timestamp": start_time,
                }


async def bounded_request(sem, session, provider, input_len, output_len, request_id):
    """Wrapper to enforce concurrency limit using Semaphore correctly."""
    async with sem:
        # Add a small jitter to avoid perfect synchronization
        await asyncio.sleep(random.uniform(0.05, 0.2))
        return await send_request(session, provider, input_len, output_len, request_id)


def save_results_incremental(results, output_file):
    """Append results to CSV file."""
    import pandas as pd

    df = pd.DataFrame(results)
    # If file doesn't exist, write header. Otherwise append without header.
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode="a", header=False, index=False)


async def main():
    """Main entry point for the latency profiling experiment."""
    if not API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please export OPENROUTER_API_KEY='your_key_here'")
        return
    try:
        import aiohttp
    except ModuleNotFoundError:
        print("Error: aiohttp is required for latency profiling.")
        print("Install the probe dependencies before running this script.")
        return

    # Setup Output File
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "experiment", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"latency_profile_llama3.3_70b_24h_{timestamp_str}.csv")
    print(f"Results will be saved incrementally to: {output_file}")

    # Semaphores
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    sem_rate_limited = asyncio.Semaphore(MAX_CONCURRENCY_RATE_LIMITED)

    # Providers
    providers_to_run = [p for p in PROVIDERS if p not in SKIP_PROVIDERS]
    if not providers_to_run:
        print("No providers to run!")
        return
    print(f"Providers to run: {providers_to_run}")

    # Experiment Loop
    total_rounds = int(LONG_RUN_CONFIG["duration_hours"] * LONG_RUN_CONFIG["rounds_per_hour"])
    round_interval = 3600 / LONG_RUN_CONFIG["rounds_per_hour"]

    print(f"Starting 24h Experiment: {total_rounds} rounds, {round_interval}s interval.")
    print(
        f"Estimated total requests per provider: {sum(w['count'] for w in LONG_RUN_CONFIG['workloads_per_round']) * total_rounds}"
    )

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for round_idx in range(total_rounds):
            round_start_time = time.time()
            print(f"\n--- Round {round_idx + 1}/{total_rounds} ---")

            tasks = []

            # Schedule tasks for all providers in this round
            for provider in providers_to_run:
                use_sem = sem_rate_limited if provider in RATE_LIMITED_PROVIDERS else sem

                for workload in LONG_RUN_CONFIG["workloads_per_round"]:
                    for i in range(workload["count"]):
                        req_id = f"{provider}-{workload['name']}-R{round_idx}-{i}"
                        task = asyncio.create_task(
                            bounded_request(
                                use_sem,
                                session,
                                provider,
                                workload["input_len"],
                                workload["output_len"],
                                req_id,
                            )
                        )
                        tasks.append(task)

            # Wait for all tasks in this round
            if tasks:
                results = await asyncio.gather(*tasks)

                # Print round stats
                success_count = sum(1 for r in results if r["status"] == "success")
                print(f"  Round finished: {success_count}/{len(results)} success.")

                # Save incrementally
                save_results_incremental(results, output_file)

            # Sleep until next round
            elapsed = time.time() - round_start_time
            sleep_time = round_interval - elapsed

            if sleep_time > 0:
                print(f"  Sleeping {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            else:
                print(f"  Warning: Round took {elapsed:.2f}s, exceeding interval {round_interval}s")

    print(f"\nExperiment Completed. Full results in {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
