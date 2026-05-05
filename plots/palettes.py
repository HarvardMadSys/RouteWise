"""Canonical color palettes for paper figures.

Conflicting palettes in the legacy code are kept under separate names so we
preserve exact pixel output of existing figures. New figures should pick one
canonical set and stick with it.
"""

from __future__ import annotations


# =============================================================================
# Provider colors (cross-cutting, used in latency profiling and provider mix)
# =============================================================================

PROVIDER_COLORS = {
    "Groq": "#2ecc71",
    "SambaNova": "#27ae60",
    "Cerebras": "#3498db",
    "Fireworks": "#e74c3c",
    "Together": "#e67e22",
    "Parasail": "#9b59b6",
    "Friendli": "#1abc9c",
    "Novita": "#f39c12",
    "Cloudflare": "#34495e",
    "Nebius": "#16a085",
    "Hyperbolic": "#8e44ad",
    "Crusoe": "#d35400",
}


# =============================================================================
# Tier colors (S_A / S_Q / S_C palette used in simulation provider-mix plots)
# =============================================================================

TIER_COLORS = {
    "quota": "#2ca02c",        # S_Q green
    "concurrency": "#9467bd",  # S_C purple
    "api": "#1f77b4",          # S_A blue
}


# =============================================================================
# Simulator policy colors
# =============================================================================

ROUTER_STRATEGY_COLORS = {
    "greedy_cost": "#1f77b4",
    "greedy_latency": "#2ca02c",
    "random": "#7f8c8d",
    "ablation_lp_only": "#d62728",
    "ablation_lp_hedging": "#ff7f0e",
    "routewise": "#9467bd",
}

ROUTER_STRATEGY_LABELS = {
    "greedy_cost": "Greedy cost",
    "greedy_latency": "Greedy latency",
    "random": "Random",
    "ablation_lp_only": "LP only",
    "ablation_lp_hedging": "LP + hedging",
    "routewise": "RouteWise",
}


# =============================================================================
# Hedging-strategy colors (Phase 4 hedging ablation: never / always / smart_*)
# =============================================================================

HEDGE_STRATEGY_COLORS = {
    "never": "#1f77b4",
    "always": "#2ca02c",
    "fixed_timeout": "#ff7f0e",
    "smart_survival": "#d62728",
    "smart_residual": "#9467bd",
}

HEDGE_STRATEGY_LABELS = {
    "never": "Never Hedge (Baseline)",
    "always": "Always Hedge",
    "fixed_timeout": "Fixed Timeout",
    "smart_survival": "Smart (Survival)",
    "smart_residual": "Smart (Residual)",
}


# =============================================================================
# Online-evaluation policy colors (Phase 5 live evaluation)
# =============================================================================

ONLINE_POLICY_COLORS = {
    "openrouter_auto": "#1f77b4",
    "sort_price": "#17becf",
    "sort_throughput": "#bcbd22",
    "sort_latency": "#e377c2",
    "lp_mix": "#2ca02c",
    "smart_hedge": "#ff7f0e",
    "cheapest_fixed": "#9467bd",
    "fastest_fixed": "#d62728",
}

ONLINE_POLICY_LABELS = {
    "openrouter_auto": "OpenRouter Auto",
    "sort_price": "sort=price",
    "sort_throughput": "sort=throughput",
    "sort_latency": "sort=latency",
    "lp_mix": "Latency-Aware (Ours)",
    "smart_hedge": "Smart Hedge (Ours)",
    "cheapest_fixed": "Cheapest Fixed",
    "fastest_fixed": "Fastest Fixed",
}


# =============================================================================
# Phase 3 routing colors (legacy)
# =============================================================================

ROUTING_COLORS = {
    "oracle": "#2ecc71",
    "lp_mix": "#3498db",
    "single_best": "#e74c3c",
    "cheapest": "#f39c12",
    "random": "#95a5a6",
}


# =============================================================================
# Lookup helpers
# =============================================================================

_FALLBACK = "#7f8c8d"


def get_provider_color(provider: str) -> str:
    """Get color for a provider, with fallback."""
    return PROVIDER_COLORS.get(provider, _FALLBACK)


def get_hedge_strategy_color(strategy: str) -> str:
    """Get color for a hedging label, with fallback (substring match)."""
    for key, color in HEDGE_STRATEGY_COLORS.items():
        if key in strategy:
            return color
    return _FALLBACK


def get_online_policy_color(policy: str) -> str:
    """Get color for an online-evaluation policy, with fallback."""
    return ONLINE_POLICY_COLORS.get(policy, _FALLBACK)
