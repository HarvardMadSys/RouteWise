"""Real online experiment harness.

Migrated from the old NSDI2027_RouteWise/experiment scripts. Provides
transport-agnostic streaming chat-completion dispatch, hedged execution,
adapter policies, and a trace-replay runner for evaluating routing
strategies against live provider APIs.

This package is the real-world counterpart to ``rwsim`` (the simulator).
The two share algorithm shapes but not implementations: real-eval uses
empirical rolling profiles where the simulator uses analytical
distributions.
"""

from experiments.real_evaluation.executor import (
    HedgedResult,
    SendFn,
    send_hedged_request,
    send_request,
)
from experiments.real_evaluation.inventory import (
    ConcurrencyState,
    InventoryConfig,
    LatencyProfile,
    PROFILE_WINDOW_SEC,
    ProviderSpec,
    ProviderState,
    QuotaState,
    build_provider_states,
    load_inventory,
)
from experiments.real_evaluation.policies import (
    BasePolicy,
    BudgetRangeHedgePolicy,
    BudgetedVhatHedgePolicy,
    CheapestFixedPolicy,
    ConcurrencyFirstPolicy,
    FastestFixedPolicy,
    OpenRouterAutoPolicy,
    OpenRouterCheapestFixedPolicy,
    OpenRouterFastestFixedPolicy,
    OriginalLPHedgePolicy,
    QuotaFirstPolicy,
    RequestContext,
    RoutingDecision,
    SortLatencyPolicy,
    SortPricePolicy,
    SortThroughputPolicy,
    build_policy,
    compute_hedge_time_sec,
    select_safe_cheapest_backup,
    slo_safe_anchor_cost,
)
from experiments.real_evaluation.recorder import (
    CSV_FIELDS,
    Recorder,
    RequestLogRow,
)
from experiments.real_evaluation.runner import (
    RealExperimentRunner,
    TraceRequest,
    load_trace_jsonl,
)
from experiments.real_evaluation.shadow_price import (
    calibrate_envelopes,
    concurrency_shadow_price,
    effective_cost,
    quota_shadow_price,
    request_marginal_cost,
)
from experiments.real_evaluation.transports import (
    BaseTransport,
    OpenAICompatStreamingTransport,
    SingleRequestResult,
    TransportConfig,
    build_transport,
    compute_request_cost_usd,
    resolve_transport_config,
)

__all__ = [
    "BasePolicy",
    "BaseTransport",
    "BudgetRangeHedgePolicy",
    "BudgetedVhatHedgePolicy",
    "CSV_FIELDS",
    "CheapestFixedPolicy",
    "ConcurrencyFirstPolicy",
    "ConcurrencyState",
    "FastestFixedPolicy",
    "HedgedResult",
    "InventoryConfig",
    "LatencyProfile",
    "OpenAICompatStreamingTransport",
    "OpenRouterAutoPolicy",
    "OpenRouterCheapestFixedPolicy",
    "OpenRouterFastestFixedPolicy",
    "OriginalLPHedgePolicy",
    "PROFILE_WINDOW_SEC",
    "ProviderSpec",
    "ProviderState",
    "QuotaFirstPolicy",
    "QuotaState",
    "RealExperimentRunner",
    "Recorder",
    "RequestContext",
    "RequestLogRow",
    "RoutingDecision",
    "SendFn",
    "SingleRequestResult",
    "SortLatencyPolicy",
    "SortPricePolicy",
    "SortThroughputPolicy",
    "TraceRequest",
    "TransportConfig",
    "build_policy",
    "build_provider_states",
    "build_transport",
    "calibrate_envelopes",
    "compute_hedge_time_sec",
    "compute_request_cost_usd",
    "concurrency_shadow_price",
    "effective_cost",
    "load_inventory",
    "load_trace_jsonl",
    "quota_shadow_price",
    "request_marginal_cost",
    "resolve_transport_config",
    "select_safe_cheapest_backup",
    "send_hedged_request",
    "send_request",
    "slo_safe_anchor_cost",
]
