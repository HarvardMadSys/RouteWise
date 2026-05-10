#!/usr/bin/env bash
set -euo pipefail

# Launch the 8h real-eval plan as one OS process per policy. Each process sees
# one Featherless account as FEATHERLESS_API_KEY and therefore one concurrency
# slot in the inventory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

TRACE="${TRACE:-data/real_eval/burstgpt_day2_h17_8h.jsonl}"
INVENTORY="${INVENTORY:-experiments/real_evaluation/data/pilot_or_chutes_subscription_featherless8_rw6_8h.json}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/real_eval/real_eval_8h_or_chutes_sub_${RUN_ID}}"
MAX_COST_USD="${MAX_COST_USD:-20}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
SPEEDUP="${SPEEDUP:-1.0}"
WARMUP_PROBES="${WARMUP_PROBES:-5}"
WARMUP_PROBE_INTERVAL_SEC="${WARMUP_PROBE_INTERVAL_SEC:-0}"
PROFILE_PROBE_SLEEP_SEC="${PROFILE_PROBE_SLEEP_SEC:-0.5}"
PERIODIC_PROBE_INTERVAL_SEC="${PERIODIC_PROBE_INTERVAL_SEC:-180}"
MIN_PROFILE_SUCCESS_SAMPLES="${MIN_PROFILE_SUCCESS_SAMPLES:-5}"
SHARED_WARMUP_PROFILE="${SHARED_WARMUP_PROFILE:-1}"
INITIAL_PROFILE_PATH="${INITIAL_PROFILE_PATH:-$OUTPUT_BASE/initial_profile.json}"

POLICIES=(
  greedy_cost
  greedy_latency
  random
  budget_range_p0_hedge
  budget_range_p25_hedge
  budget_range_p50_hedge
  budget_range_p75_hedge
  budget_range_p100_hedge
)

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required in $ENV_FILE or the environment" >&2
  exit 2
fi

FEATHERLESS_KEYS=()
if [[ -n "${FEATHERLESS_API_KEYS:-}" ]]; then
  IFS=',' read -r -a FEATHERLESS_KEYS <<< "$FEATHERLESS_API_KEYS"
else
  for n in 1 2 3 4 5 6 7 8; do
    var="FEATHERLESS_API_KEY_${n}"
    value="${!var:-}"
    if [[ -n "$value" ]]; then
      FEATHERLESS_KEYS+=("$value")
    fi
  done
fi

if [[ "${#FEATHERLESS_KEYS[@]}" -ne "${#POLICIES[@]}" ]]; then
  echo "expected ${#POLICIES[@]} Featherless keys from FEATHERLESS_API_KEYS or FEATHERLESS_API_KEY_1..8, got ${#FEATHERLESS_KEYS[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_BASE"
printf '%s\n' "${POLICIES[@]}" > "$OUTPUT_BASE/policies.txt"

PROCESS_WARMUP_PROBES="$WARMUP_PROBES"
INITIAL_PROFILE_ARGS=()
if [[ "$SHARED_WARMUP_PROFILE" != "0" && "$WARMUP_PROBES" -gt 0 ]]; then
  echo "prebuilding shared initial profile -> $INITIAL_PROFILE_PATH"
  FEATHERLESS_API_KEY="${FEATHERLESS_KEYS[0]}" uv run python scripts/prebuild_profile.py \
    --inventory "$INVENTORY" \
    --output "$INITIAL_PROFILE_PATH" \
    --probes-per-provider "$WARMUP_PROBES" \
    --profile-probe-sleep-sec "$PROFILE_PROBE_SLEEP_SEC" \
    --round-interval-sec "$WARMUP_PROBE_INTERVAL_SEC" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-cost-usd "$MAX_COST_USD"
  PROCESS_WARMUP_PROBES=0
  INITIAL_PROFILE_ARGS=(--initial-profile-path "$INITIAL_PROFILE_PATH")
fi

cat > "$OUTPUT_BASE/run_env.txt" <<EOF
TRACE=$TRACE
INVENTORY=$INVENTORY
MAX_COST_USD=$MAX_COST_USD
TIMEOUT_SEC=$TIMEOUT_SEC
SPEEDUP=$SPEEDUP
WARMUP_PROBES=$WARMUP_PROBES
PROCESS_WARMUP_PROBES=$PROCESS_WARMUP_PROBES
SHARED_WARMUP_PROFILE=$SHARED_WARMUP_PROFILE
INITIAL_PROFILE_PATH=$INITIAL_PROFILE_PATH
WARMUP_PROBE_INTERVAL_SEC=$WARMUP_PROBE_INTERVAL_SEC
PROFILE_PROBE_SLEEP_SEC=$PROFILE_PROBE_SLEEP_SEC
PERIODIC_PROBE_INTERVAL_SEC=$PERIODIC_PROBE_INTERVAL_SEC
MIN_PROFILE_SUCCESS_SAMPLES=$MIN_PROFILE_SUCCESS_SAMPLES
EOF

pids=()
for i in "${!POLICIES[@]}"; do
  policy="${POLICIES[$i]}"
  key="${FEATHERLESS_KEYS[$i]}"
  out="$OUTPUT_BASE/$policy"
  mkdir -p "$out"
  echo "launching $policy -> $out"
  (
    FEATHERLESS_API_KEY="$key" uv run python -m experiments.real_evaluation \
      --inventory "$INVENTORY" \
      --trace "$TRACE" \
      --policy "$policy" \
      --output "$out" \
      --speedup "$SPEEDUP" \
      --max-cost-usd "$MAX_COST_USD" \
      --timeout-sec "$TIMEOUT_SEC" \
      --warmup-probes "$PROCESS_WARMUP_PROBES" \
      "${INITIAL_PROFILE_ARGS[@]}" \
      --warmup-probe-interval-sec "$WARMUP_PROBE_INTERVAL_SEC" \
      --profile-probe-sleep-sec "$PROFILE_PROBE_SLEEP_SEC" \
      --periodic-probe-interval-sec "$PERIODIC_PROBE_INTERVAL_SEC" \
      --min-profile-success-samples "$MIN_PROFILE_SUCCESS_SAMPLES" \
      > "$out/run.log" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "one or more policy processes failed; inspect $OUTPUT_BASE/*/run.log" >&2
  exit 1
fi

echo "all policy processes completed: $OUTPUT_BASE"
