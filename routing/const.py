"""
Constants for the waiting queue API and outsourcing engine.
"""

# ============================================================================
# Pricing Constants
# ============================================================================

# Default pricing per token (in dollars)
DEFAULT_INPUT_PRICE_PER_TOKEN = 1.25 / 1_000_000  # $1.25 per million input tokens
DEFAULT_OUTPUT_PRICE_PER_TOKEN = 10.0 / 1_000_000  # $10.0 per million output tokens

# ============================================================================
# Outsourcing Engine Constants
# ============================================================================

# Default configuration for OutsourcingEngine
DEFAULT_MAX_BATCH_SIZE = 256
DEFAULT_PREFILL_SLO_BASE_SECONDS = 4.05
DEFAULT_PREFILL_SLO_SLACK_FACTOR = 10.0
DEFAULT_UTILIZATION_TARGET = 0.8

# ============================================================================
# Model Architecture Constants
# ============================================================================

# Default transformer model parameters
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_NUM_LAYERS = 32
DEFAULT_NUM_ATTENTION_HEADS = 32
DEFAULT_DEVICE_TFLOPS = 312.0  # A100 GPU peak performance

# ============================================================================
# FLOP Calculation Constants
# ============================================================================

# FLOP multipliers for transformer operations
ATTENTION_FLOP_MULTIPLIER = 2  # For QK^T + softmax * V operations
FFN_FLOP_MULTIPLIER = 4  # For two linear projections in FFN

# Unit conversions
TFLOPS_TO_FLOPS = 1e12

# ============================================================================
# Numerical Constants
# ============================================================================

# Small values for numerical stability
MIN_WEIGHT_VALUE = 1.0
MIN_VALUE_FOR_RATIO = 1e-10

# ============================================================================
# Tree Cache Constants
# ============================================================================

# Tree Cache Configuration
MAX_TREE_NODE_LENGTH = 128
MAX_TREE_CACHE_SIZE = 100_000_000
