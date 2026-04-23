#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE="${ENV_FILE:-}"
TRACE_FILE="${TRACE_FILE:-${PROJECT_ROOT}/experiment/data/day3_trace.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/experiment/results/phase6_sa_online_apr23_us}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
WARMUP="${WARMUP:-300}"
PROBING_INTERVAL="${PROBING_INTERVAL:-300}"
COST_CAP="${COST_CAP:-10}"
SLO="${SLO:-2.0}"
TAU="${TAU:-0.75}"
REFERENCE_MODE="${REFERENCE_MODE:-max_provider}"
SEED="${SEED:-42}"
REAL_PROMPTS="${REAL_PROMPTS:-1}"
MODEL_COOLDOWN="${MODEL_COOLDOWN:-300}"

if [[ -n "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

declare -a MODELS=(
  "deepseek/deepseek-v3.2"
  "qwen/qwen3-235b-a22b-2507"
  "meta-llama/llama-3.3-70b-instruct"
)

declare -A MODEL_PRICING=(
  ["deepseek/deepseek-v3.2"]="${PROJECT_ROOT}/experiment/data/openrouter_deepseek_v32.json"
  ["qwen/qwen3-235b-a22b-2507"]="${PROJECT_ROOT}/experiment/data/openrouter_qwen3_235b_2507.json"
  ["meta-llama/llama-3.3-70b-instruct"]="${PROJECT_ROOT}/experiment/data/openrouter_llama33_70b.json"
)

declare -a POLICIES=(
  "openrouter_auto"
  "sort_price"
  "sort_throughput"
  "sort_latency"
  "cheapest_fixed"
  "fastest_fixed"
  "lp_mix"
  "smart_hedge"
  "budget_vhat_t75_hedge_explorer"
)

echo "============================================================"
echo "Phase 6 S_A-only online matrix"
echo "============================================================"
echo "Run root: ${RUN_ROOT}"
echo "Trace: ${TRACE_FILE}"
echo "Time limit: ${TIME_LIMIT}s"
echo "Warmup: ${WARMUP}s"
echo "Probing interval: ${PROBING_INTERVAL}s"
echo "Cost cap: ${COST_CAP}"
echo "SLO: ${SLO}s"
echo "Tau: ${TAU}"
echo "Reference mode: ${REFERENCE_MODE}"
echo "Model cooldown: ${MODEL_COOLDOWN}s"
echo "Policies: ${POLICIES[*]}"
echo "Models: ${MODELS[*]}"
echo "Execution: one shared-prober coordinator per model"
echo "============================================================"

for model_idx in "${!MODELS[@]}"; do
  model="${MODELS[${model_idx}]}"
  model_slug="$(echo "${model}" | tr '/:.-' '_')"
  model_log_dir="${LOG_ROOT}/${model_slug}"
  mkdir -p "${model_log_dir}"

  pricing_file="${MODEL_PRICING[${model}]}"
  if [[ ! -f "${pricing_file}" ]]; then
    echo "Missing pricing file for ${model}: ${pricing_file}" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Starting model batch: ${model}"
  echo "Pricing: ${pricing_file}"
  echo "Logs: ${model_log_dir}"
  echo "============================================================"

  log_file="${model_log_dir}/shared_prober.log"
  cmd=(
    "${PYTHON_BIN}"
    "${PROJECT_ROOT}/experiment/scripts/phase6_sa_online_evaluation.py"
    --model "${model}"
    --policies "${POLICIES[@]}"
    --pricing "${pricing_file}"
    --trace "${TRACE_FILE}"
    --output "${RUN_ROOT}"
    --slo "${SLO}"
    --warmup "${WARMUP}"
    --probing-interval "${PROBING_INTERVAL}"
    --cost-cap "${COST_CAP}"
    --time-limit "${TIME_LIMIT}"
    --tau "${TAU}"
    --reference-mode "${REFERENCE_MODE}"
    --seed "${SEED}"
  )

  if [[ "${REAL_PROMPTS}" == "1" ]]; then
    cmd+=(--real-prompts)
  fi

  echo "[launch] ${model} :: shared-prober coordinator"
  if (
    cd "${PROJECT_ROOT}"
    "${cmd[@]}"
  ) >"${log_file}" 2>&1; then
    echo "[done] ${model} :: shared-prober coordinator"
  else
    echo "[failed] ${model} :: shared-prober coordinator (see ${log_file})" >&2
    exit 1
  fi

  if [[ "${model_idx}" -lt "$((${#MODELS[@]} - 1))" && "${MODEL_COOLDOWN}" -gt 0 ]]; then
    echo "[cooldown] sleeping ${MODEL_COOLDOWN}s before next model"
    sleep "${MODEL_COOLDOWN}"
  fi
done

echo
echo "All model batches completed successfully."
echo "Results: ${RUN_ROOT}"
