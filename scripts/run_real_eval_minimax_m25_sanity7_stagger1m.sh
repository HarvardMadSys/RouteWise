#!/usr/bin/env bash
set -euo pipefail

# MiniMax M2.5 real-eval sanity run after the hedge-cancel profile fix:
# - same BurstGPT cap10s trace and MiniMax OR+subscription inventory as the
#   previous 8h/alpha-sweep run
# - seven sanity policies: OR baselines, greedy baselines, p75 hedge/no-hedge
# - one minute launch stagger across every policy
#
# Override DURATION_SEC=3600 for a 1h run. The default is 2h.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${TRACE:=data/real_eval/burstgpt_day0_24h_cap10s.jsonl}"
if [[ ! -f "$TRACE" ]]; then
  echo "error: trace not found: $TRACE" >&2
  echo "regenerate it first (see README, Data section):" >&2
  echo "  python3 scripts/prepare_workload.py --start-day 0 --days 1 --output data/real_eval/burstgpt_day0_24h.jsonl" >&2
  echo "  python3 scripts/idle_compress_trace.py --source data/real_eval/burstgpt_day0_24h.jsonl --output data/real_eval/burstgpt_day0_24h_cap10s.jsonl" >&2
  exit 1
fi
: "${INVENTORY:=experiments/real_evaluation/data/pilot_or_minimax_subscription_or8_top_24h.json}"
: "${OUTPUT_BASE:=outputs/real_eval/real_eval_minimax_m25_fixed_cancel_sanity7_stagger1m_${RUN_ID}}"

: "${POLICY_LIST:=or_auto or_sort_latency or_sort_cost greedy_latency greedy_cost budget_range_alpha75_hedge budget_range_alpha75}"
: "${STAGGER_SEC:=60}"
: "${STAGGER_ALL_POLICIES:=1}"

: "${QUOTA_PROVIDER:=none}"
: "${OPENROUTER_KEY_MODE:=single}"
: "${PREFIX_CACHE_ROUTING:=1}"

: "${SLO_MS:=3000}"
: "${DURATION_SEC:=7200}"
: "${SPEEDUP:=1.0}"
: "${TIMEOUT_SEC:=60}"
: "${MAX_COST_USD:=80}"

: "${WARMUP_PROBES:=24}"
: "${SHARED_WARMUP_PROFILE:=1}"
: "${SHARED_PROFILE_PROBING:=1}"
: "${MIN_PROFILE_SUCCESS_SAMPLES:=5}"

# The current Featherless setup has four policy accounts, where slots 1-3 each
# have 4 units and slot 4 has 8 units. MiniMax M2.5 consumes 4 units/request,
# so slot 4 can safely back two logical policy slots. If KEY5 exists, reserve it
# for shared profile probing and expose key4 twice for the policy slots.
_read_dotenv_value() {
  local key="$1"
  local line value

  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" .env | tail -n 1 || true)"
  [[ -n "$line" ]] || return 1

  line="${line#"${line%%[![:space:]]*}"}"
  line="${line#export }"
  line="${line#export	}"
  value="${line#*=}"
  value="${value%$'\r'}"

  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi

  printf '%s' "$value"
}

if [[ -f .env ]]; then
  for key in FEATHERLESS_API_KEYS FEATHERLESS_API_KEY1 FEATHERLESS_API_KEY2 FEATHERLESS_API_KEY3 FEATHERLESS_API_KEY4 FEATHERLESS_API_KEY5; do
    if [[ -z "${!key:-}" ]]; then
      value="$(_read_dotenv_value "$key" || true)"
      if [[ -n "$value" ]]; then
        export "$key=$value"
      fi
    fi
  done
fi
if [[ -z "${FEATHERLESS_API_KEYS:-}" \
  && -n "${FEATHERLESS_API_KEY1:-}" \
  && -n "${FEATHERLESS_API_KEY2:-}" \
  && -n "${FEATHERLESS_API_KEY3:-}" \
  && -n "${FEATHERLESS_API_KEY4:-}" ]]; then
  if [[ -n "${FEATHERLESS_API_KEY5:-}" ]]; then
    export FEATHERLESS_API_KEYS="${FEATHERLESS_API_KEY5},${FEATHERLESS_API_KEY1},${FEATHERLESS_API_KEY2},${FEATHERLESS_API_KEY3},${FEATHERLESS_API_KEY4},${FEATHERLESS_API_KEY4}"
  else
    export FEATHERLESS_API_KEYS="${FEATHERLESS_API_KEY1},${FEATHERLESS_API_KEY2},${FEATHERLESS_API_KEY3},${FEATHERLESS_API_KEY4},${FEATHERLESS_API_KEY4}"
  fi
fi

export RUN_ID
export TRACE
export INVENTORY
export OUTPUT_BASE
export POLICY_LIST
export STAGGER_SEC
export STAGGER_ALL_POLICIES
export QUOTA_PROVIDER
export OPENROUTER_KEY_MODE
export PREFIX_CACHE_ROUTING
export SLO_MS
export DURATION_SEC
export SPEEDUP
export TIMEOUT_SEC
export MAX_COST_USD
export WARMUP_PROBES
export SHARED_WARMUP_PROFILE
export SHARED_PROFILE_PROBING
export MIN_PROFILE_SUCCESS_SAMPLES

exec scripts/run_real_eval_8h_policy_processes.sh
