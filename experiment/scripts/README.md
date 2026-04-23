# Experiment Scripts

This directory contains scripts for running offline routing experiments and generating analysis.

## Scripts

### 1. `run_simulation.py`

Run offline routing simulation with different strategies.

**Usage:**
```bash
# Run with default config (Optimal strategy)
python experiment/scripts/run_simulation.py

# Run with specific config
python experiment/scripts/run_simulation.py \
  --config config/experiment.yaml \
  --output experiment/results/my_results.json

# Run with different number of subscriptions
python experiment/scripts/run_simulation.py \
  --num-subscriptions 2 \
  --output experiment/results/2subs.json
```

**Strategies:**
- Optimal: Offline optimal with perfect future knowledge
- All-API: Baseline without subscription
- Greedy: Simple online algorithm (use quota first)

### 2. `generate_plots.py`

Generate publication-quality figures from experiment results.

**Usage:**
```bash
python experiment/scripts/generate_plots.py
```

**Output:**
- `experiment/results/figures/fig1_cost_comparison.pdf/png`
- `experiment/results/figures/fig2_cost_breakdown.pdf/png`
- `experiment/results/figures/fig3_quota_utilization.pdf/png`
- `experiment/results/figures/fig4_savings_analysis.pdf/png`
- `experiment/results/figures/fig5_competitive_ratio.pdf/png`

### 3. `compare_results.py`

Compare multiple experiment results.

**Usage:**
```bash
python experiment/scripts/compare_results.py \
  experiment/results/*.json
```

### 4. `analyze_actual_vs_optimal.py`

Analyze actual vs optimal routing decisions.

**Usage:**
```bash
python experiment/scripts/analyze_actual_vs_optimal.py
```

## Quick Start

### Run Complete Experiment

```bash
# 1. Run All-API baseline
python experiment/scripts/run_simulation.py \
  --config config/experiment.yaml \
  --strategy all-api \
  --output experiment/results/chatgpt_all_api.json

# 2. Run Greedy
python experiment/scripts/run_simulation.py \
  --config config/experiment.yaml \
  --strategy greedy \
  --output experiment/results/chatgpt_greedy.json

# 3. Run Optimal
python experiment/scripts/run_simulation.py \
  --config config/experiment.yaml \
  --output experiment/results/chatgpt_optimal.json

# 4. Generate plots
python experiment/scripts/generate_plots.py
```

## Results

Results are saved to `experiment/results/`:
- JSON files with detailed metrics
- `figures/` directory with publication-quality plots

## Configuration

Experiments use config file from `config/`:
- `experiment.yaml` - Main experiment configuration (currently single model: ChatGPT)

To test different models, modify the `filter_model` and `model_mapping` in the config.

## For OSDI Submission

The generated figures are publication-ready:
- PDF format for LaTeX papers
- PNG format for presentations
- 300 DPI resolution
- Clean, professional styling

## Ollama Cloud Quota Probe

To estimate a conservative 5-hour session lower bound for an Ollama Cloud plan
under a fixed workload, use:

```bash
source .venv/bin/activate
export OLLAMA_API_KEY=your_api_key

python scripts/perf/ollama_quota_probe.py \
  --model glm-4.7:cloud \
  --prompt "Summarize why tail latency matters in LLM serving." \
  --output-dir experiment/results/ollama_quota_probe/pro_run1
```

Recommended setup:
- Prefer `glm-4.7:cloud` or `minimax-m2.5:cloud` for coding-heavy RouteWise
  experiments. Avoid mixing model families across repeated quota runs.
- For a clean per-workload calibration pass, use `--concurrency 1`.
- To burn budget faster and estimate plan-level limits, use `--concurrency 3`
  (the documented Ollama Pro cloud concurrency limit).
- Keep `temperature = 0` and a fixed prompt set for repeatability.
- Run one full session window and record the number of successful requests
  before the first usage-limit error.
- Repeat across multiple 5-hour windows and take the minimum observed success
  count as the safe lower bound for that workload.

More accurate plan-limit estimation:

```bash
source .venv/bin/activate
export OLLAMA_API_KEY=your_api_key

python scripts/perf/ollama_quota_probe.py \
  --model glm-4.7:cloud \
  --prompt-file prompts.jsonl \
  --concurrency 3 \
  --max-duration-minutes 120 \
  --session-start-pct 0.0 \
  --session-end-pct 7.4 \
  --weekly-start-pct 0.0 \
  --weekly-end-pct 1.1 \
  --output-dir experiment/results/ollama_quota_probe/pro_block1
```

The script will derive:
- implied 5-hour request-equivalent budget
- implied weekly request-equivalent budget
- conservative lower bounds based on dashboard quantization and per-request P95/P99 usage
