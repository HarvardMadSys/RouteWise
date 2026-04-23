"""Data loaders for various datasets."""

import asyncio
import csv
import time
from collections.abc import AsyncGenerator

from serving.base import LLMRequest

from .base import DataLoader


class BurstGPTLoader(DataLoader):
    """Loader for BurstGPT dataset with time-based replay."""

    def __init__(self, trace_file: str, time_scale: float = 100.0, max_requests: int | None = None):
        """
        Args:
            trace_file: Path to BurstGPT CSV trace file
            time_scale: Speed multiplier (100 = 100x faster than original)
            max_requests: Limit number of requests
        """
        self.trace_file = trace_file
        self.time_scale = time_scale
        self.max_requests = max_requests

    def _get_prompt(self, target_tokens: int) -> str:
        """Generate a prompt with approximately target token count."""
        return f"Please generate a comprehensive prompt or query that would be approximately {target_tokens} tokens long. The prompt should be realistic and cover a complex topic requiring detailed analysis."

    async def load_requests(self) -> AsyncGenerator[LLMRequest, None]:
        """Load and replay requests from BurstGPT trace."""
        print(f"Loading BurstGPT trace from {self.trace_file}")
        print(f"Time scale: {self.time_scale}x (original time / {self.time_scale})")

        requests = []
        count = 0

        # Parse CSV trace
        with open(self.trace_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.max_requests and count >= self.max_requests:
                    break

                timestamp = float(row["Timestamp"])
                model = row["Model"].lower().replace("-", "_")
                request_tokens = int(row["Request tokens"])

                prompt = self._get_prompt(request_tokens)

                request = LLMRequest(
                    prompt=prompt,
                    model=model,
                    max_tokens=min(
                        int(row["Response tokens"]), 512
                    ),  # Use expected response length
                    temperature=0.7,
                )

                requests.append((timestamp, request))
                count += 1

        print(f"Loaded {len(requests)} requests")

        # Sort by timestamp
        requests.sort(key=lambda x: x[0])

        # Replay with timing
        if not requests:
            return

        start_time = time.time()
        first_timestamp = requests[0][0]

        for i, (timestamp, request) in enumerate(requests):
            # Calculate when to send this request
            relative_seconds = (timestamp - first_timestamp) / self.time_scale
            target_time = start_time + relative_seconds

            # Wait if needed
            current_time = time.time()
            if target_time > current_time:
                await asyncio.sleep(target_time - current_time)

            if i % 100 == 0:
                print(f"Sent {i + 1}/{len(requests)} requests")

            yield request


class SplitwiseLoader(DataLoader):
    """Loader for Splitwise/Vidur trace format."""

    def __init__(self, trace_file: str, time_scale: float = 1.0, max_requests: int | None = None):
        """
        Args:
            trace_file: Path to Splitwise CSV trace file
            time_scale: Speed multiplier (1 = original speed)
            max_requests: Limit number of requests
        """
        self.trace_file = trace_file
        self.time_scale = time_scale
        self.max_requests = max_requests

    async def load_requests(self) -> AsyncGenerator[LLMRequest, None]:
        """Load and replay requests from Splitwise trace.

        Format: arrived_at, num_prefill_tokens, num_decode_tokens
        """
        print(f"Loading Splitwise trace from {self.trace_file}")
        print(f"Time scale: {self.time_scale}x")

        requests = []

        # Parse CSV trace
        with open(self.trace_file) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if self.max_requests and i >= self.max_requests:
                    break

                timestamp = float(row["arrived_at"])
                prefill_tokens = int(row["num_prefill_tokens"])
                decode_tokens = int(row["num_decode_tokens"])

                # Simple prompt based on token count
                prompt = f"Generate text with context length of {prefill_tokens} tokens"

                request = LLMRequest(
                    prompt=prompt,
                    model="",  # Will be overridden
                    max_tokens=decode_tokens,
                    temperature=0.7,
                )

                requests.append((timestamp, request))

        print(f"Loaded {len(requests)} requests")

        if not requests:
            return

        # Replay with timing
        start_time = asyncio.get_event_loop().time()

        for i, (timestamp, request) in enumerate(requests):
            # Calculate when to send
            target_time = timestamp / self.time_scale
            current_time = asyncio.get_event_loop().time() - start_time

            if target_time > current_time:
                await asyncio.sleep(target_time - current_time)

            if (i + 1) % 100 == 0:
                elapsed = asyncio.get_event_loop().time() - start_time
                print(f"[{elapsed:.1f}s] Sent {i + 1}/{len(requests)} requests")

            yield request
