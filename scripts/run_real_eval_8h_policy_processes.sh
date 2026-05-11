#!/usr/bin/env bash
set -euo pipefail

# Launch the 8h real-eval plan as one OS process per policy. Joint-pool
# policies get one Featherless account and one OpenRouter key each. Native
# OpenRouter baselines share OPENROUTER_API_KEY_1 so they measure OR behavior
# without consuming the per-policy key pool.

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

DEFAULT_POLICY_LIST="greedy_cost greedy_latency random budget_range_p0_hedge budget_range_p25_hedge budget_range_p50_hedge budget_range_p75_hedge budget_range_p100_hedge or_auto or_sort_latency or_sort_cost"
# Override with POLICY_LIST="..." when running a smaller or alternate set.
read -r -a POLICIES <<< "${POLICY_LIST:-$DEFAULT_POLICY_LIST}"

is_native_or_baseline() {
  case "$1" in
    or_auto|or_sort_latency|or_sort_cost|or_sort_throughput)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

requires_featherless_key() {
  case "$1" in
    or_auto|or_sort_latency|or_sort_cost|or_sort_throughput|or_greedy_cost|or_greedy_latency)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

# Joint-pool policies that route to Chutes_SQ need a dedicated Chutes key
# (subscription-tier direct API). OR-only baselines do not.
requires_chutes_key() {
  case "$1" in
    or_auto|or_sort_latency|or_sort_cost|or_sort_throughput|or_greedy_cost|or_greedy_latency)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

OPENROUTER_KEYS=()
if [[ -n "${OPENROUTER_API_KEYS:-}" ]]; then
  IFS=',' read -r -a OPENROUTER_KEYS <<< "$OPENROUTER_API_KEYS"
else
  saw_numbered_openrouter_key=0
  for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
    var="OPENROUTER_API_KEY_${n}"
    alt_var="OPENROUTER_API_KEY${n}"
    value="${!var:-${!alt_var:-}}"
    if [[ -n "$value" ]]; then
      saw_numbered_openrouter_key=1
      OPENROUTER_KEYS+=("$value")
    fi
  done
  if [[ "$saw_numbered_openrouter_key" -eq 0 && -n "${OPENROUTER_API_KEY:-}" ]]; then
    OPENROUTER_KEYS+=("$OPENROUTER_API_KEY")
  fi
fi

if [[ "${#OPENROUTER_KEYS[@]}" -eq 0 ]]; then
  echo "OPENROUTER_API_KEY, OPENROUTER_API_KEYS, or OPENROUTER_API_KEY_1..N is required in $ENV_FILE or the environment" >&2
  exit 2
fi

FEATHERLESS_KEYS=()
if [[ -n "${FEATHERLESS_API_KEYS:-}" ]]; then
  IFS=',' read -r -a FEATHERLESS_KEYS <<< "$FEATHERLESS_API_KEYS"
else
  for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
    var="FEATHERLESS_API_KEY_${n}"
    alt_var="FEATHERLESS_API_KEY${n}"
    value="${!var:-${!alt_var:-}}"
    if [[ -n "$value" ]]; then
      FEATHERLESS_KEYS+=("$value")
    fi
  done
  if [[ "${#FEATHERLESS_KEYS[@]}" -eq 0 && -n "${FEATHERLESS_API_KEY:-}" ]]; then
    FEATHERLESS_KEYS+=("$FEATHERLESS_API_KEY")
  fi
fi

CHUTES_KEYS=()
if [[ -n "${CHUTES_API_KEYS:-}" ]]; then
  IFS=',' read -r -a CHUTES_KEYS <<< "$CHUTES_API_KEYS"
else
  for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
    var="CHUTES_API_KEY_${n}"
    alt_var="CHUTES_API_KEY${n}"
    value="${!var:-${!alt_var:-}}"
    if [[ -n "$value" ]]; then
      CHUTES_KEYS+=("$value")
    fi
  done
  if [[ "${#CHUTES_KEYS[@]}" -eq 0 && -n "${CHUTES_API_KEY:-}" ]]; then
    CHUTES_KEYS+=("$CHUTES_API_KEY")
  fi
fi

FEATHERLESS_POLICY_COUNT=0
CHUTES_POLICY_COUNT=0
OPENROUTER_DEDICATED_POLICY_COUNT=0
HAS_NATIVE_OR_BASELINE=0
for policy in "${POLICIES[@]}"; do
  if requires_featherless_key "$policy"; then
    FEATHERLESS_POLICY_COUNT=$((FEATHERLESS_POLICY_COUNT + 1))
  fi
  if requires_chutes_key "$policy"; then
    CHUTES_POLICY_COUNT=$((CHUTES_POLICY_COUNT + 1))
  fi
  if is_native_or_baseline "$policy"; then
    HAS_NATIVE_OR_BASELINE=1
  else
    OPENROUTER_DEDICATED_POLICY_COUNT=$((OPENROUTER_DEDICATED_POLICY_COUNT + 1))
  fi
done

if [[ "${#FEATHERLESS_KEYS[@]}" -lt "$FEATHERLESS_POLICY_COUNT" ]]; then
  echo "expected at least $FEATHERLESS_POLICY_COUNT Featherless keys from FEATHERLESS_API_KEYS or FEATHERLESS_API_KEY_1..N, got ${#FEATHERLESS_KEYS[@]}" >&2
  exit 2
fi

if [[ "$CHUTES_POLICY_COUNT" -gt 0 && "${#CHUTES_KEYS[@]}" -lt "$CHUTES_POLICY_COUNT" ]]; then
  echo "expected at least $CHUTES_POLICY_COUNT Chutes keys from CHUTES_API_KEYS or CHUTES_API_KEY_1..N, got ${#CHUTES_KEYS[@]}" >&2
  exit 2
fi

OR_KEYS_REQUIRED=$OPENROUTER_DEDICATED_POLICY_COUNT
if [[ "$HAS_NATIVE_OR_BASELINE" -eq 1 ]]; then
  OR_KEYS_REQUIRED=$((OR_KEYS_REQUIRED + 1))
fi
if [[ "${#OPENROUTER_KEYS[@]}" -lt "$OR_KEYS_REQUIRED" ]]; then
  echo "expected at least $OR_KEYS_REQUIRED OpenRouter keys: key1 for native OR baselines plus one key per non-native policy; got ${#OPENROUTER_KEYS[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_BASE"
printf '%s\n' "${POLICIES[@]}" > "$OUTPUT_BASE/policies.txt"
ASSIGNMENTS_PATH="$OUTPUT_BASE/policy_key_assignments.tsv"
printf 'policy\topenrouter_key_slot\tfeatherless_key_slot\tchutes_key_slot\tstart_delay_sec\n' > "$ASSIGNMENTS_PATH"

# Stagger configuration. OR-only baselines never staggered (they use OR's
# server-side routing and don't contend for our pinned providers). Other
# policies (joint-pool / RouteWise / random / greedy) start STAGGER_SEC apart
# so their concurrent bursts onto pinned providers are spread out per
# Juncheng's recommendation.
STAGGER_SEC="${STAGGER_SEC:-0}"

PROCESS_WARMUP_PROBES="$WARMUP_PROBES"
INITIAL_PROFILE_ARGS=()
if [[ "$SHARED_WARMUP_PROFILE" != "0" && "$WARMUP_PROBES" -gt 0 ]]; then
  echo "prebuilding shared initial profile -> $INITIAL_PROFILE_PATH"
  CHUTES_API_KEY="${CHUTES_KEYS[0]:-}" FEATHERLESS_API_KEY="${FEATHERLESS_KEYS[0]:-}" OPENROUTER_API_KEY="${OPENROUTER_KEYS[0]}" uv run python scripts/prebuild_profile.py \
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
POLICY_LIST=${POLICIES[*]}
OPENROUTER_KEY_COUNT=${#OPENROUTER_KEYS[@]}
FEATHERLESS_KEY_COUNT=${#FEATHERLESS_KEYS[@]}
FEATHERLESS_POLICY_COUNT=$FEATHERLESS_POLICY_COUNT
EOF

pids=()
featherless_idx=0
chutes_idx=0
dedicated_or_idx=1
non_or_launch_idx=0
if [[ "$HAS_NATIVE_OR_BASELINE" -eq 0 ]]; then
  dedicated_or_idx=0
fi
for i in "${!POLICIES[@]}"; do
  policy="${POLICIES[$i]}"
  featherless_key=""
  featherless_key_slot=""
  if requires_featherless_key "$policy"; then
    featherless_key="${FEATHERLESS_KEYS[$featherless_idx]}"
    featherless_key_slot=$((featherless_idx + 1))
    featherless_idx=$((featherless_idx + 1))
  fi
  chutes_key=""
  chutes_key_slot=""
  if requires_chutes_key "$policy" && [[ "${#CHUTES_KEYS[@]}" -gt 0 ]]; then
    chutes_key="${CHUTES_KEYS[$chutes_idx]}"
    chutes_key_slot=$((chutes_idx + 1))
    chutes_idx=$((chutes_idx + 1))
  fi
  if is_native_or_baseline "$policy"; then
    openrouter_key="${OPENROUTER_KEYS[0]}"
    openrouter_key_slot=1
    start_delay=0
  else
    openrouter_key="${OPENROUTER_KEYS[$dedicated_or_idx]}"
    openrouter_key_slot=$((dedicated_or_idx + 1))
    dedicated_or_idx=$((dedicated_or_idx + 1))
    start_delay=$((non_or_launch_idx * STAGGER_SEC))
    non_or_launch_idx=$((non_or_launch_idx + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$policy" "$openrouter_key_slot" "$featherless_key_slot" "$chutes_key_slot" "$start_delay" >> "$ASSIGNMENTS_PATH"
  out="$OUTPUT_BASE/$policy"
  mkdir -p "$out"
  echo "launching $policy -> $out (start_delay=${start_delay}s)"
  (
    if [[ "$start_delay" -gt 0 ]]; then
      sleep "$start_delay"
    fi
    CHUTES_API_KEY="$chutes_key" FEATHERLESS_API_KEY="$featherless_key" OPENROUTER_API_KEY="$openrouter_key" uv run python -m experiments.real_evaluation \
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
