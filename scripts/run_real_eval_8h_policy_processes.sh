#!/usr/bin/env bash
set -euo pipefail

# Launch the 8h real-eval plan as one OS process per policy. By default every
# process uses OPENROUTER_API_KEY_1 so provider failures and billing state are
# comparable across policies. Set OPENROUTER_KEY_MODE=per_policy to opt into
# the older one-OpenRouter-key-per-joint-policy layout for rate-limit stress
# tests.
#
# Quota tier (Chutes vs MiniMax native) is selected by QUOTA_PROVIDER:
#   chutes  (default) — inject CHUTES_API_KEY for joint-pool policies
#   minimax           — inject MINIMAX_API_KEY for joint-pool policies (one key
#                       is enough; multiple joint policies round-robin keys and
#                       share quota if the key is the same account)
#   none              — do not inject a native quota key; use inventory/env as-is
#                       (e.g. OpenRouter-emulated quota or ZAI_API_KEY)
# If QUOTA_PROVIDER is empty, it defaults from INVENTORY path (*pilot_minimax*
# or *minimax_plus_direct* -> minimax, else chutes).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -n "$line" && "${line:0:1}" != "#" ]] || continue

    if [[ "$line" == export[[:space:]]* ]]; then
      line="${line#export}"
      line="${line#"${line%%[![:space:]]*}"}"
    fi

    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue

    key="${line%%=*}"
    value="${line#*=}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ -z "${!key:-}" ]]; then
      export "$key=$value"
    fi
  done < "$ENV_FILE"
fi

TRACE="${TRACE:-data/real_eval/burstgpt_day2_h17_8h.jsonl}"
INVENTORY="${INVENTORY:-experiments/real_evaluation/data/pilot_chutes_direct_or8_top_8h.json}"
# Optional comma-separated overrides:
#   POLICY_INVENTORY_MAP="greedy_cost=path_a.json,random=path_a.json,budget_range_p75_hedge=path_b.json"
# Policies not listed here use INVENTORY.
POLICY_INVENTORY_MAP="${POLICY_INVENTORY_MAP:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/real_eval/real_eval_8h_direct_or8_top_${RUN_ID}}"
MAX_COST_USD="${MAX_COST_USD:-20}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
SPEEDUP="${SPEEDUP:-1.0}"
QUOTA_WINDOW_ANCHOR="${QUOTA_WINDOW_ANCHOR:-wall_clock}"
# Optional inventory-SLO override. When set, passes --slo-ms to every runner
# process. Useful for SLO ablations without editing the inventory file.
SLO_MS="${SLO_MS:-}"
# Optional wall-clock cap on the replay phase. Defensive: if a policy hangs
# the runner stops dispatching new trace requests after this many seconds.
# Leave unset to let the runner replay the whole trace without a cap.
DURATION_SEC="${DURATION_SEC:-}"
WARMUP_PROBES="${WARMUP_PROBES:-24}"
WARMUP_PROBE_INTERVAL_SEC="${WARMUP_PROBE_INTERVAL_SEC:-5}"
PROFILE_PROBE_SLEEP_SEC="${PROFILE_PROBE_SLEEP_SEC:-0.5}"
MIN_PROFILE_SUCCESS_SAMPLES="${MIN_PROFILE_SUCCESS_SAMPLES:-5}"
SHARED_WARMUP_PROFILE="${SHARED_WARMUP_PROFILE:-1}"
INITIAL_PROFILE_PATH="${INITIAL_PROFILE_PATH:-$OUTPUT_BASE/initial_profile.json}"
# Real-eval runs launch one process per policy, so natural/probe feedback is
# shared across processes by default. The optional sidecar below adds active
# probes; turning it off should not disable natural-feedback sharing.
SHARED_PROFILE_PROBING="${SHARED_PROFILE_PROBING:-1}"
SHARED_PROFILE_EVENTS="${SHARED_PROFILE_EVENTS:-1}"
SHARED_PROFILE_EVENTS_PATH="${SHARED_PROFILE_EVENTS_PATH:-$OUTPUT_BASE/shared_profile_events.jsonl}"
SHARED_PROFILE_POLL_SEC="${SHARED_PROFILE_POLL_SEC:-1}"
SHARED_PROFILE_PROBE_INTERVAL_SEC="${SHARED_PROFILE_PROBE_INTERVAL_SEC:-5}"
SHARED_PROFILE_IDLE_SEC="${SHARED_PROFILE_IDLE_SEC:-5}"
SHARED_PROFILE_MAX_PROBES_PER_TICK="${SHARED_PROFILE_MAX_PROBES_PER_TICK:-0}"
SYNTHESIZE_MISSING_PROMPTS="${SYNTHESIZE_MISSING_PROMPTS:-0}"
SYNTHETIC_OUTPUT_TOKENS="${SYNTHETIC_OUTPUT_TOKENS:-}"
PREFIX_CACHE_ROUTING="${PREFIX_CACHE_ROUTING:-0}"
KEY_SLOT_MAX="${KEY_SLOT_MAX:-20}"
OPENROUTER_KEY_MODE="${OPENROUTER_KEY_MODE:-single}"
if [[ "$OPENROUTER_KEY_MODE" != "single" && "$OPENROUTER_KEY_MODE" != "per_policy" ]]; then
  echo "OPENROUTER_KEY_MODE must be single or per_policy; got: $OPENROUTER_KEY_MODE" >&2
  exit 2
fi
if [[ "$QUOTA_WINDOW_ANCHOR" != "wall_clock" && "$QUOTA_WINDOW_ANCHOR" != "trace_start" ]]; then
  echo "QUOTA_WINDOW_ANCHOR must be wall_clock or trace_start; got: $QUOTA_WINDOW_ANCHOR" >&2
  exit 2
fi
if [[ "$SHARED_PROFILE_PROBING" != "0" && "$SHARED_PROFILE_EVENTS" == "0" ]]; then
  echo "SHARED_PROFILE_PROBING requires SHARED_PROFILE_EVENTS=1" >&2
  exit 2
fi

# chutes | minimax — which native subscription API keys joint-pool policies get
if [[ -z "${QUOTA_PROVIDER:-}" ]]; then
  case "${INVENTORY:-}" in
    *pilot_minimax*|*minimax_plus_direct*)
      QUOTA_PROVIDER=minimax
      ;;
    *)
      QUOTA_PROVIDER=chutes
      ;;
  esac
fi
if [[ "$QUOTA_PROVIDER" != "chutes" && "$QUOTA_PROVIDER" != "minimax" && "$QUOTA_PROVIDER" != "none" ]]; then
  echo "QUOTA_PROVIDER must be chutes, minimax, or none; got: $QUOTA_PROVIDER" >&2
  exit 2
fi

DEFAULT_POLICY_LIST="greedy_cost greedy_latency random budget_range_p0_hedge budget_range_p25_hedge budget_range_p50_hedge budget_range_p75_hedge budget_range_p100_hedge or_auto or_sort_latency or_sort_cost"
# Override with POLICY_LIST="..." when running a smaller or alternate set.
read -r -a POLICIES <<< "${POLICY_LIST:-$DEFAULT_POLICY_LIST}"

if [[ ! -f "$TRACE" ]]; then
  echo "TRACE file not found: $TRACE" >&2
  echo "Set TRACE to an existing JSONL workload before starting real eval." >&2
  exit 2
fi
if [[ ! -f "$INVENTORY" ]]; then
  echo "INVENTORY file not found: $INVENTORY" >&2
  exit 2
fi

if [[ -n "$POLICY_INVENTORY_MAP" ]]; then
  IFS=',' read -r -a _POLICY_INVENTORY_PAIRS <<< "$POLICY_INVENTORY_MAP"
  for pair in "${_POLICY_INVENTORY_PAIRS[@]}"; do
    policy_name="${pair%%=*}"
    policy_inventory="${pair#*=}"
    if [[ -z "$policy_name" || -z "$policy_inventory" || "$policy_name" == "$policy_inventory" ]]; then
      echo "invalid POLICY_INVENTORY_MAP entry: $pair" >&2
      exit 2
    fi
  done
fi

inventory_for_policy() {
  local policy="$1"
  local pair policy_name policy_inventory
  if [[ -n "$POLICY_INVENTORY_MAP" ]]; then
    IFS=',' read -r -a _POLICY_INVENTORY_PAIRS <<< "$POLICY_INVENTORY_MAP"
    for pair in "${_POLICY_INVENTORY_PAIRS[@]}"; do
      policy_name="${pair%%=*}"
      policy_inventory="${pair#*=}"
      if [[ "$policy_name" == "$policy" ]]; then
        printf '%s' "$policy_inventory"
        return 0
      fi
    done
  fi
  printf '%s' "$INVENTORY"
}

for policy in "${POLICIES[@]}"; do
  policy_inventory="$(inventory_for_policy "$policy")"
  if [[ ! -f "$policy_inventory" ]]; then
    echo "inventory file not found for policy $policy: $policy_inventory" >&2
    exit 2
  fi
done

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

# Joint-pool policies that use the inventory quota tier (Chutes_SQ or
# MiniMax_Plus_SQ, etc.) need a dedicated native API key. OR-only baselines do not.
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
  for n in $(seq 1 "$KEY_SLOT_MAX"); do
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
  for n in $(seq 1 "$KEY_SLOT_MAX"); do
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
  for n in $(seq 1 "$KEY_SLOT_MAX"); do
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

MINIMAX_KEYS=()
if [[ -n "${MINIMAX_API_KEYS:-}" ]]; then
  IFS=',' read -r -a MINIMAX_KEYS <<< "$MINIMAX_API_KEYS"
else
  for n in $(seq 1 "$KEY_SLOT_MAX"); do
    var="MINIMAX_API_KEY_${n}"
    alt_var="MINIMAX_API_KEY${n}"
    value="${!var:-${!alt_var:-}}"
    if [[ -n "$value" ]]; then
      MINIMAX_KEYS+=("$value")
    fi
  done
  if [[ "${#MINIMAX_KEYS[@]}" -eq 0 && -n "${MINIMAX_API_KEY:-}" ]]; then
    MINIMAX_KEYS+=("$MINIMAX_API_KEY")
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

# When the shared profile prober runs, reserve FEATHERLESS_KEYS[0] for it so
# warmup + sidecar probes never compete with real-policy traffic for the
# single concurrency slot on that account. Policies then draw from index 1
# onward. With active probing disabled, the only up-front probes are one-shot
# warmup probes, which complete before any policy starts, so sharing index 0 is
# safe.
FEATHERLESS_POLICY_START_IDX=0
if [[ "$SHARED_PROFILE_PROBING" != "0" ]]; then
  FEATHERLESS_POLICY_START_IDX=1
fi
FEATHERLESS_REQUIRED=$((FEATHERLESS_POLICY_COUNT + FEATHERLESS_POLICY_START_IDX))

if [[ "${#FEATHERLESS_KEYS[@]}" -lt "$FEATHERLESS_REQUIRED" ]]; then
  if [[ "$FEATHERLESS_POLICY_START_IDX" -gt 0 ]]; then
    echo "expected at least $FEATHERLESS_REQUIRED Featherless keys ($FEATHERLESS_POLICY_COUNT joint policies + 1 dedicated to the active shared profile prober) from FEATHERLESS_API_KEYS or FEATHERLESS_API_KEY_1..N, got ${#FEATHERLESS_KEYS[@]}. Set SHARED_PROFILE_PROBING=0 to disable active shared probing and let policies use the first key; shared profile events remain enabled unless SHARED_PROFILE_EVENTS=0 is set explicitly." >&2
  else
    echo "expected at least $FEATHERLESS_POLICY_COUNT Featherless keys from FEATHERLESS_API_KEYS or FEATHERLESS_API_KEY_1..N, got ${#FEATHERLESS_KEYS[@]}" >&2
  fi
  exit 2
fi

if [[ "$QUOTA_PROVIDER" == "none" ]]; then
  :
elif [[ "$QUOTA_PROVIDER" == "minimax" ]]; then
  if [[ "$CHUTES_POLICY_COUNT" -gt 0 && "${#MINIMAX_KEYS[@]}" -eq 0 ]]; then
    echo "QUOTA_PROVIDER=minimax requires at least one of MINIMAX_API_KEY, MINIMAX_API_KEYS, or MINIMAX_API_KEY_1..N in $ENV_FILE" >&2
    exit 2
  fi
  if [[ "$CHUTES_POLICY_COUNT" -gt 0 && "${#MINIMAX_KEYS[@]}" -gt 0 && "${#MINIMAX_KEYS[@]}" -lt "$CHUTES_POLICY_COUNT" ]]; then
    echo "WARNING: $CHUTES_POLICY_COUNT joint policies; only ${#MINIMAX_KEYS[@]} MiniMax key(s). Keys are assigned in round-robin. If all keys are the same account, every process shares one MiniMax Plus quota (4500 requests / 5h)." >&2
  fi
else
  if [[ "$CHUTES_POLICY_COUNT" -gt 0 && "${#CHUTES_KEYS[@]}" -lt "$CHUTES_POLICY_COUNT" ]]; then
    # Not strictly an error: inventory may use openrouter transport for
    # Chutes_SQ (no direct Chutes key needed). Warn so users notice when they
    # ARE expecting direct Chutes; assignment cycles through available keys
    # and policies past the cycle get an empty CHUTES_API_KEY.
    echo "WARNING: $CHUTES_POLICY_COUNT joint policies; only ${#CHUTES_KEYS[@]} Chutes keys. OK if inventory routes Chutes via OpenRouter (transport=openrouter). Otherwise direct Chutes calls past key #${#CHUTES_KEYS[@]} will fail with missing CHUTES_API_KEY." >&2
  fi
fi

OR_KEYS_REQUIRED=1
if [[ "$OPENROUTER_KEY_MODE" == "per_policy" ]]; then
  OR_KEYS_REQUIRED=$OPENROUTER_DEDICATED_POLICY_COUNT
  if [[ "$HAS_NATIVE_OR_BASELINE" -eq 1 ]]; then
    OR_KEYS_REQUIRED=$((OR_KEYS_REQUIRED + 1))
  fi
fi
if [[ "${#OPENROUTER_KEYS[@]}" -lt "$OR_KEYS_REQUIRED" ]]; then
  if [[ "$OPENROUTER_KEY_MODE" == "per_policy" ]]; then
    # Allow running with fewer keys than the "ideal" 1-per-non-native-policy
    # layout. Keys are round-robin-assigned below, so concurrent policies will
    # share OR per-key rate limits and may see extra 429s.
    echo "WARNING: $OR_KEYS_REQUIRED OpenRouter keys recommended (key1 for native OR baselines + one per non-native policy); got ${#OPENROUTER_KEYS[@]}. Concurrent policies will share keys and may hit shared OR rate limits." >&2
  else
    echo "ERROR: OPENROUTER_KEY_MODE=single requires OPENROUTER_API_KEY_1, OPENROUTER_API_KEY1, OPENROUTER_API_KEYS, or OPENROUTER_API_KEY" >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT_BASE"
printf '%s\n' "${POLICIES[@]}" > "$OUTPUT_BASE/policies.txt"
ASSIGNMENTS_PATH="$OUTPUT_BASE/policy_key_assignments.tsv"
printf 'policy\topenrouter_key_slot\tfeatherless_key_slot\tquota_native_key_slot\tstart_delay_sec\tinventory\n' > "$ASSIGNMENTS_PATH"

# Stagger configuration. OR-only baselines never staggered (they use OR's
# server-side routing and don't contend for our pinned providers). Other
# policies (joint-pool / RouteWise / random / greedy) start STAGGER_SEC apart
# so their concurrent bursts onto pinned providers are spread out per
# Juncheng's recommendation.
STAGGER_SEC="${STAGGER_SEC:-30}"
STAGGER_ALL_POLICIES="${STAGGER_ALL_POLICIES:-0}"

PREBUILD_SHARED_PROFILE_ARGS=()
if [[ "$SHARED_PROFILE_EVENTS" != "0" ]]; then
  mkdir -p "$(dirname "$SHARED_PROFILE_EVENTS_PATH")"
  : > "$SHARED_PROFILE_EVENTS_PATH"
  PREBUILD_SHARED_PROFILE_ARGS=(--shared-profile-events "$SHARED_PROFILE_EVENTS_PATH")
fi

PROCESS_WARMUP_PROBES="$WARMUP_PROBES"
INITIAL_PROFILE_ARGS=()
if [[ "$SHARED_WARMUP_PROFILE" != "0" && "$WARMUP_PROBES" -gt 0 ]]; then
  echo "prebuilding shared initial profile -> $INITIAL_PROFILE_PATH"
  prebuild_chutes_key=""
  prebuild_minimax_key=""
  if [[ "$QUOTA_PROVIDER" == "minimax" ]]; then
    prebuild_minimax_key="${MINIMAX_KEYS[0]:-}"
  elif [[ "$QUOTA_PROVIDER" == "chutes" ]]; then
    prebuild_chutes_key="${CHUTES_KEYS[0]:-}"
  fi
  CHUTES_API_KEY="$prebuild_chutes_key" MINIMAX_API_KEY="$prebuild_minimax_key" FEATHERLESS_API_KEY="${FEATHERLESS_KEYS[0]:-}" OPENROUTER_API_KEY="${OPENROUTER_KEYS[0]}" uv run python scripts/prebuild_profile.py \
  --inventory "$INVENTORY" \
  --output "$INITIAL_PROFILE_PATH" \
  --probes-per-provider "$WARMUP_PROBES" \
  --profile-probe-sleep-sec "$PROFILE_PROBE_SLEEP_SEC" \
  --round-interval-sec "$WARMUP_PROBE_INTERVAL_SEC" \
  --timeout-sec "$TIMEOUT_SEC" \
  --max-cost-usd "$MAX_COST_USD" \
  "${PREBUILD_SHARED_PROFILE_ARGS[@]}"
  PROCESS_WARMUP_PROBES=0
  INITIAL_PROFILE_ARGS=(--initial-profile-path "$INITIAL_PROFILE_PATH")
fi

TRACE_ARGS=()
if [[ "$SYNTHESIZE_MISSING_PROMPTS" != "0" ]]; then
  TRACE_ARGS+=(--synthesize-missing-prompts)
  if [[ -n "$SYNTHETIC_OUTPUT_TOKENS" ]]; then
    TRACE_ARGS+=(--synthetic-output-tokens "$SYNTHETIC_OUTPUT_TOKENS")
  fi
fi
if [[ "$PREFIX_CACHE_ROUTING" != "0" ]]; then
  TRACE_ARGS+=(--prefix-cache-routing)
fi

EXTRA_RUNNER_ARGS=()
if [[ -n "$SLO_MS" ]]; then
  EXTRA_RUNNER_ARGS+=(--slo-ms "$SLO_MS")
fi
if [[ -n "$DURATION_SEC" ]]; then
  EXTRA_RUNNER_ARGS+=(--duration-sec "$DURATION_SEC")
fi
EXTRA_RUNNER_ARGS+=(--quota-window-anchor "$QUOTA_WINDOW_ANCHOR")

cat > "$OUTPUT_BASE/run_env.txt" <<EOF
TRACE=$TRACE
INVENTORY=$INVENTORY
POLICY_INVENTORY_MAP=$POLICY_INVENTORY_MAP
QUOTA_PROVIDER=$QUOTA_PROVIDER
MAX_COST_USD=$MAX_COST_USD
TIMEOUT_SEC=$TIMEOUT_SEC
SPEEDUP=$SPEEDUP
QUOTA_WINDOW_ANCHOR=$QUOTA_WINDOW_ANCHOR
SLO_MS=$SLO_MS
DURATION_SEC=$DURATION_SEC
STAGGER_SEC=$STAGGER_SEC
STAGGER_ALL_POLICIES=$STAGGER_ALL_POLICIES
WARMUP_PROBES=$WARMUP_PROBES
PROCESS_WARMUP_PROBES=$PROCESS_WARMUP_PROBES
SHARED_WARMUP_PROFILE=$SHARED_WARMUP_PROFILE
INITIAL_PROFILE_PATH=$INITIAL_PROFILE_PATH
WARMUP_PROBE_INTERVAL_SEC=$WARMUP_PROBE_INTERVAL_SEC
PROFILE_PROBE_SLEEP_SEC=$PROFILE_PROBE_SLEEP_SEC
SHARED_PROFILE_PROBING=$SHARED_PROFILE_PROBING
SHARED_PROFILE_EVENTS=$SHARED_PROFILE_EVENTS
SHARED_PROFILE_EVENTS_PATH=$SHARED_PROFILE_EVENTS_PATH
SHARED_PROFILE_POLL_SEC=$SHARED_PROFILE_POLL_SEC
SHARED_PROFILE_PROBE_INTERVAL_SEC=$SHARED_PROFILE_PROBE_INTERVAL_SEC
SHARED_PROFILE_IDLE_SEC=$SHARED_PROFILE_IDLE_SEC
SHARED_PROFILE_MAX_PROBES_PER_TICK=$SHARED_PROFILE_MAX_PROBES_PER_TICK
MIN_PROFILE_SUCCESS_SAMPLES=$MIN_PROFILE_SUCCESS_SAMPLES
SYNTHESIZE_MISSING_PROMPTS=$SYNTHESIZE_MISSING_PROMPTS
SYNTHETIC_OUTPUT_TOKENS=$SYNTHETIC_OUTPUT_TOKENS
PREFIX_CACHE_ROUTING=$PREFIX_CACHE_ROUTING
KEY_SLOT_MAX=$KEY_SLOT_MAX
OPENROUTER_KEY_MODE=$OPENROUTER_KEY_MODE
POLICY_LIST=${POLICIES[*]}
OPENROUTER_KEY_COUNT=${#OPENROUTER_KEYS[@]}
FEATHERLESS_KEY_COUNT=${#FEATHERLESS_KEYS[@]}
FEATHERLESS_POLICY_COUNT=$FEATHERLESS_POLICY_COUNT
CHUTES_KEY_COUNT=${#CHUTES_KEYS[@]}
MINIMAX_KEY_COUNT=${#MINIMAX_KEYS[@]}
EOF

SHARED_PROFILE_ARGS=()
profile_probe_pid=""
if [[ "$SHARED_PROFILE_EVENTS" != "0" ]]; then
  mkdir -p "$(dirname "$SHARED_PROFILE_EVENTS_PATH")"
  touch "$SHARED_PROFILE_EVENTS_PATH"
  SHARED_PROFILE_ARGS=(
    --shared-profile-events "$SHARED_PROFILE_EVENTS_PATH"
    --shared-profile-poll-sec "$SHARED_PROFILE_POLL_SEC"
  )
fi
if [[ "$SHARED_PROFILE_PROBING" != "0" ]]; then
  prebuild_chutes_key=""
  prebuild_minimax_key=""
  if [[ "$QUOTA_PROVIDER" == "minimax" ]]; then
    prebuild_minimax_key="${MINIMAX_KEYS[0]:-}"
  elif [[ "$QUOTA_PROVIDER" == "chutes" ]]; then
    prebuild_chutes_key="${CHUTES_KEYS[0]:-}"
  fi
  echo "launching shared profile prober -> $OUTPUT_BASE/shared_profile_probe.log"
  CHUTES_API_KEY="$prebuild_chutes_key" MINIMAX_API_KEY="$prebuild_minimax_key" FEATHERLESS_API_KEY="${FEATHERLESS_KEYS[0]:-}" OPENROUTER_API_KEY="${OPENROUTER_KEYS[0]}" uv run python scripts/shared_profile_probe.py \
    --inventory "$INVENTORY" \
    --shared-profile-events "$SHARED_PROFILE_EVENTS_PATH" \
    --interval-sec "$SHARED_PROFILE_PROBE_INTERVAL_SEC" \
    --idle-threshold-sec "$SHARED_PROFILE_IDLE_SEC" \
    --max-probes-per-tick "$SHARED_PROFILE_MAX_PROBES_PER_TICK" \
    --profile-probe-sleep-sec "$PROFILE_PROBE_SLEEP_SEC" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-cost-usd "$MAX_COST_USD" \
    > "$OUTPUT_BASE/shared_profile_probe.log" 2>&1 &
  profile_probe_pid="$!"
  sleep 2
  if ! kill -0 "$profile_probe_pid" 2>/dev/null; then
    echo "ERROR: shared profile prober died on startup; see $OUTPUT_BASE/shared_profile_probe.log" >&2
    tail -n 20 "$OUTPUT_BASE/shared_profile_probe.log" >&2 || true
    exit 1
  fi
fi

pids=()
cleanup_children() {
  local code=$?
  if [[ -n "${profile_probe_pid:-}" ]]; then
    kill "$profile_probe_pid" 2>/dev/null || true
  fi
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [[ -n "${profile_probe_pid:-}" ]]; then
    wait "$profile_probe_pid" 2>/dev/null || true
  fi
  for pid in "${pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$code"
}
trap cleanup_children INT TERM

featherless_idx=$FEATHERLESS_POLICY_START_IDX
quota_native_idx=0
dedicated_or_idx=1
non_or_launch_idx=0
all_policy_launch_idx=0
if [[ "$HAS_NATIVE_OR_BASELINE" -eq 0 ]]; then
  dedicated_or_idx=0
fi
for i in "${!POLICIES[@]}"; do
  policy="${POLICIES[$i]}"
  policy_inventory="$(inventory_for_policy "$policy")"
  featherless_key=""
  featherless_key_slot=""
  if requires_featherless_key "$policy"; then
    featherless_key="${FEATHERLESS_KEYS[$featherless_idx]}"
    featherless_key_slot=$((featherless_idx + 1))
    featherless_idx=$((featherless_idx + 1))
  fi
  native_quota_key=""
  native_quota_key_slot=""
  if requires_chutes_key "$policy"; then
    if [[ "$QUOTA_PROVIDER" == "minimax" ]]; then
      if [[ "${#MINIMAX_KEYS[@]}" -gt 0 ]]; then
        nmk="${#MINIMAX_KEYS[@]}"
        native_quota_key="${MINIMAX_KEYS[$((quota_native_idx % nmk))]}"
        native_quota_key_slot=$(( (quota_native_idx % nmk) + 1 ))
        quota_native_idx=$((quota_native_idx + 1))
      fi
    else
      if [[ "${#CHUTES_KEYS[@]}" -gt 0 && "$quota_native_idx" -lt "${#CHUTES_KEYS[@]}" ]]; then
        native_quota_key="${CHUTES_KEYS[$quota_native_idx]}"
        native_quota_key_slot=$((quota_native_idx + 1))
        quota_native_idx=$((quota_native_idx + 1))
      fi
    fi
  fi
  if [[ "$STAGGER_ALL_POLICIES" != "0" ]]; then
    start_delay=$((all_policy_launch_idx * STAGGER_SEC))
    all_policy_launch_idx=$((all_policy_launch_idx + 1))
  elif is_native_or_baseline "$policy"; then
    start_delay=0
  else
    start_delay=$((non_or_launch_idx * STAGGER_SEC))
    non_or_launch_idx=$((non_or_launch_idx + 1))
  fi
  if [[ "$OPENROUTER_KEY_MODE" == "single" ]]; then
    openrouter_key="${OPENROUTER_KEYS[0]}"
    openrouter_key_slot=1
  elif is_native_or_baseline "$policy"; then
    openrouter_key="${OPENROUTER_KEYS[0]}"
    openrouter_key_slot=1
  else
    # Round-robin into the available key pool so the script works with any
    # number of keys (1 .. N). When #keys < #non-native policies the assigned
    # key is shared with another concurrent policy.
    or_pool_size="${#OPENROUTER_KEYS[@]}"
    or_idx=$((dedicated_or_idx % or_pool_size))
    openrouter_key="${OPENROUTER_KEYS[$or_idx]}"
    openrouter_key_slot=$((or_idx + 1))
    dedicated_or_idx=$((dedicated_or_idx + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$policy" "$openrouter_key_slot" "$featherless_key_slot" "$native_quota_key_slot" "$start_delay" "$policy_inventory" >> "$ASSIGNMENTS_PATH"
  out="$OUTPUT_BASE/$policy"
  mkdir -p "$out"
  echo "launching $policy -> $out (start_delay=${start_delay}s)"
  (
    if [[ "$start_delay" -gt 0 ]]; then
      sleep "$start_delay"
    fi
    chutes_for_run=""
    minimax_for_run=""
    if [[ "$QUOTA_PROVIDER" == "minimax" ]]; then
      minimax_for_run="$native_quota_key"
    elif [[ "$QUOTA_PROVIDER" == "chutes" ]]; then
      chutes_for_run="$native_quota_key"
    fi
    CHUTES_API_KEY="$chutes_for_run" MINIMAX_API_KEY="$minimax_for_run" FEATHERLESS_API_KEY="$featherless_key" OPENROUTER_API_KEY="$openrouter_key" uv run python -m experiments.real_evaluation \
      --inventory "$policy_inventory" \
      --trace "$TRACE" \
      --policy "$policy" \
      --output "$out" \
      --speedup "$SPEEDUP" \
      --max-cost-usd "$MAX_COST_USD" \
      --timeout-sec "$TIMEOUT_SEC" \
      --warmup-probes "$PROCESS_WARMUP_PROBES" \
      "${INITIAL_PROFILE_ARGS[@]}" \
      "${SHARED_PROFILE_ARGS[@]}" \
      "${TRACE_ARGS[@]}" \
      "${EXTRA_RUNNER_ARGS[@]}" \
      --warmup-probe-interval-sec "$WARMUP_PROBE_INTERVAL_SEC" \
      --profile-probe-sleep-sec "$PROFILE_PROBE_SLEEP_SEC" \
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
trap - INT TERM

if [[ -n "$profile_probe_pid" ]]; then
  kill "$profile_probe_pid" 2>/dev/null || true
  wait "$profile_probe_pid" 2>/dev/null || true
fi

if [[ "$failed" -ne 0 ]]; then
  echo "one or more policy processes failed; inspect $OUTPUT_BASE/*/run.log" >&2
  exit 1
fi

echo "all policy processes completed: $OUTPUT_BASE"
