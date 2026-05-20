#!/usr/bin/env bash
set -euo pipefail

# MiniMax M2.5 main real-eval run:
# - one 24h BurstGPT trace compressed with the cap10s workload file
# - seven paper-facing policies
# - 10 minute launch stagger across every policy
#
# The cap10s trace spans about 8.1h of replay time. DURATION_SEC=32400 is
# intentionally long enough to replay the full compressed trace while leaving
# headroom for slow requests. This is the MiniMax counterpart of the GLM-5.1
# main7 script.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"
: "${TRACE:=data/real_eval/burstgpt_day0_24h_cap10s.jsonl}"
: "${INVENTORY:=experiments/real_evaluation/data/pilot_or_minimax_subscription_or8_top_24h.json}"
: "${OUTPUT_BASE:=outputs/real_eval/real_eval_minimax_m25_burstgpt_cap10s_main7_stagger10m_${RUN_ID}}"

: "${POLICY_LIST:=budget_range_p75_hedge budget_range_p75 greedy_latency or_sort_latency or_auto or_sort_cost greedy_cost}"
: "${STAGGER_SEC:=600}"
: "${STAGGER_ALL_POLICIES:=1}"

: "${QUOTA_PROVIDER:=none}"
: "${OPENROUTER_KEY_MODE:=single}"
: "${PREFIX_CACHE_ROUTING:=1}"

: "${SLO_MS:=3000}"
: "${DURATION_SEC:=32400}"
: "${SPEEDUP:=1.0}"
: "${TIMEOUT_SEC:=60}"
: "${MAX_COST_USD:=250}"

: "${WARMUP_PROBES:=24}"
: "${SHARED_WARMUP_PROFILE:=1}"
: "${SHARED_PROFILE_EVENTS:=1}"
: "${SHARED_PROFILE_PROBING:=1}"
: "${PERIODIC_PROBE_INTERVAL_SEC:=180}"
: "${MIN_PROFILE_SUCCESS_SAMPLES:=5}"

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
  for key in FEATHERLESS_API_KEYS FEATHERLESS_API_KEY1 FEATHERLESS_API_KEY2 FEATHERLESS_API_KEY3 FEATHERLESS_API_KEY4; do
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
  export FEATHERLESS_API_KEYS="${FEATHERLESS_API_KEY1},${FEATHERLESS_API_KEY2},${FEATHERLESS_API_KEY3},${FEATHERLESS_API_KEY4},${FEATHERLESS_API_KEY4}"
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
export SHARED_PROFILE_EVENTS
export SHARED_PROFILE_PROBING
export PERIODIC_PROBE_INTERVAL_SEC
export MIN_PROFILE_SUCCESS_SAMPLES

exec scripts/run_real_eval_8h_policy_processes.sh
