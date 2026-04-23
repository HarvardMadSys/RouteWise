#!/bin/bash
# DEPRECATED — do not use.
#
# This launcher is retained as documentation of the old 24-hour closed-
# loop experiment. It pre-dates the following methodology fixes (all on
# the claude/exciting-perlman branch):
#
#   e0530d9  fastest_fixed removed, per-policy routers, SMART_ECONOMIC
#            uses real C_b/V, load_pricing parameterized
#   01973ad  open-loop ThreadPoolExecutor dispatch, --wall-clock-limit,
#            P1 mode fail-fast, probing cadence matches paper
#   af495ea  bounded in-flight semaphore, background probing thread,
#            checkpoint uses dispatched_entries not task count,
#            total_cost lock promoted to instance attribute
#   59df126  mid-entry abort cannot mark a partial entry as dispatched,
#            probing_loop's update_lp() holds the router lock
#
# Running this script today would silently exercise the old harness
# (inline probing stalls, unbounded queue, checkpoint counter bug,
# contaminated router state) and produce results that are NOT
# publication-safe. External review specifically flagged this main-repo
# launcher as a rerun blocker because an operator running it from the
# main repo would hit the stale phase5_online_evaluation.py here
# instead of the reviewed copy in the worktree.
#
# The correct replacement is
#   .claude/worktrees/exciting-perlman/experiment/scripts/run_7d_qwen3.sh
# which pins an absolute path to the reviewed phase5 script and
# preflights a fingerprint grep before starting.

cat <<'DEPRECATED_MSG' >&2
==============================================================================
DEPRECATED: run_24h_burstgpt.sh (main repo copy)

This launcher predates the round-1..4 NSDI methodology fixes. Running it
would silently exercise the old (pre-fix) phase5_online_evaluation.py in
the main repo, not the reviewed copy in the worktree. Results would NOT
be publication-safe.

Use instead:
  bash /home/murphy/test/hybridInference/.claude/worktrees/exciting-perlman/experiment/scripts/run_7d_qwen3.sh check

See run_24h_burstgpt.sh header comment for the commits that made this
launcher obsolete.
==============================================================================
DEPRECATED_MSG
exit 1

# ---------- Unreachable legacy body kept for reference only ----------

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
RESULTS_BASE="$PROJECT_ROOT/experiment/results/phase5_online"

# Experiment parameters
# Use JSONL file with real prompts (P1 mode)
TRACE_FILE="$PROJECT_ROOT/data/sharegpt_burstgpt/sharegpt_prompts_burstgpt_timestamps.jsonl"
TIME_LIMIT=86400  # 24 hours in seconds
COST_CAP=100.0    # $100 budget (higher for real prompts)
CHECKPOINT_INTERVAL=100
WARMUP=300        # 5 min warmup
PROBING_INTERVAL=300  # 5 min probing
SPEEDUP=1.0       # Real-time replay
SLO=2.0           # 2 second SLO
MAX_OUTPUT_TOKENS=256  # Cap output tokens to control cost

# All 5 policies for comprehensive comparison
POLICIES="openrouter_auto lp_mix smart_hedge cheapest_fixed fastest_fixed"

# Generate run ID based on timestamp
generate_run_id() {
    echo "run_$(date +%Y%m%d_%H%M%S)"
}

# Find the latest run directory (with PID file, indicating it was started by this script)
find_latest_run() {
    for dir in $(ls -td "$RESULTS_BASE"/run_* 2>/dev/null); do
        if [ -f "$dir/pid" ]; then
            echo "$dir"
            return
        fi
    done
    # Fallback to most recent if no PID file found
    ls -td "$RESULTS_BASE"/run_* 2>/dev/null | head -1
}

# Start new experiment
start_experiment() {
    RUN_ID=$(generate_run_id)
    RUN_DIR="$RESULTS_BASE/$RUN_ID"
    LOG_FILE="$RUN_DIR/nohup.log"
    PID_FILE="$RUN_DIR/pid"

    mkdir -p "$RUN_DIR"

    echo "=============================================="
    echo "Starting 24h BurstGPT Experiment"
    echo "=============================================="
    echo "Run ID: $RUN_ID"
    echo "Output: $RUN_DIR"
    echo "Trace: $TRACE_FILE"
    echo "Time limit: $TIME_LIMIT sec (24h)"
    echo "Cost cap: \$$COST_CAP"
    echo "Policies: $POLICIES"
    echo "=============================================="

    # Activate venv and run
    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"

    # Use unbuffered output (-u) so log updates in real-time
    nohup python -u experiment/scripts/phase5_online_evaluation.py \
        --trace "$TRACE_FILE" \
        --time-limit "$TIME_LIMIT" \
        --cost-cap "$COST_CAP" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --warmup "$WARMUP" \
        --probing-interval "$PROBING_INTERVAL" \
        --speedup "$SPEEDUP" \
        --slo "$SLO" \
        --real-prompts \
        --max-output-tokens "$MAX_OUTPUT_TOKENS" \
        --policies $POLICIES \
        --output "$RESULTS_BASE" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"

    echo ""
    echo "Experiment started in background!"
    echo "PID: $(cat "$PID_FILE")"
    echo ""
    echo "Monitor commands:"
    echo "  $0 status  - Check progress"
    echo "  $0 tail    - Follow live output"
    echo "  $0 stop    - Stop experiment"
    echo ""
    echo "Log file: $LOG_FILE"
}

# Resume from checkpoint
resume_experiment() {
    RESUME_DIR="${1:-$(find_latest_run)}"

    if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
        echo "Error: No run directory found to resume"
        echo "Usage: $0 resume [run_directory]"
        exit 1
    fi

    LOG_FILE="$RESUME_DIR/nohup.log"
    PID_FILE="$RESUME_DIR/pid"

    echo "=============================================="
    echo "Resuming Experiment from Checkpoint"
    echo "=============================================="
    echo "Resume dir: $RESUME_DIR"
    echo "=============================================="

    # Check if already running
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "Error: Experiment already running (PID: $OLD_PID)"
            exit 1
        fi
    fi

    cd "$PROJECT_ROOT"
    source "$VENV_PATH/bin/activate"

    # Use unbuffered output (-u) so log updates in real-time
    nohup python -u experiment/scripts/phase5_online_evaluation.py \
        --trace "$TRACE_FILE" \
        --time-limit "$TIME_LIMIT" \
        --cost-cap "$COST_CAP" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --warmup "$WARMUP" \
        --probing-interval "$PROBING_INTERVAL" \
        --speedup "$SPEEDUP" \
        --slo "$SLO" \
        --real-prompts \
        --max-output-tokens "$MAX_OUTPUT_TOKENS" \
        --policies $POLICIES \
        --resume-dir "$RESUME_DIR" \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"

    echo ""
    echo "Experiment resumed!"
    echo "PID: $(cat "$PID_FILE")"
}

# Check status
check_status() {
    RUN_DIR="${1:-$(find_latest_run)}"

    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "No experiment found"
        exit 1
    fi

    echo "=============================================="
    echo "Experiment Status"
    echo "=============================================="
    echo "Run dir: $RUN_DIR"
    echo ""

    # Check if running
    PID_FILE="$RUN_DIR/pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Status: RUNNING (PID: $PID)"
            RUNTIME=$(ps -p "$PID" -o etime= 2>/dev/null || echo "unknown")
            echo "Runtime: $RUNTIME"
        else
            echo "Status: STOPPED"
        fi
    else
        echo "Status: UNKNOWN (no PID file)"
    fi
    echo ""

    # Check checkpoint
    CHECKPOINT="$RUN_DIR/checkpoint.pkl"
    if [ -f "$CHECKPOINT" ]; then
        echo "Checkpoint:"
        python3 -c "
from pathlib import Path
import pickle
import datetime
with open('$CHECKPOINT', 'rb') as f:
    cp = pickle.load(f)
print(f'  Completed requests: {cp.completed_requests}')
print(f'  Total cost: \${cp.total_cost:.4f}')
print(f'  Last save: {datetime.datetime.fromtimestamp(cp.last_save_time).strftime(\"%Y-%m-%d %H:%M:%S\")}')
for p, c in cp.policy_costs.items():
    print(f'  {p}: \${c:.4f}')
" 2>/dev/null || echo "  (unable to read checkpoint)"
    fi
    echo ""

    # Check CSV log (may be in RUN_DIR or a sibling directory created by Python)
    CSV_LOG="$RUN_DIR/evaluation_log.csv"
    if [ ! -f "$CSV_LOG" ]; then
        # Python creates its own timestamped dir, check siblings
        for sibling in $(ls -td "$RESULTS_BASE"/run_* 2>/dev/null | head -5); do
            if [ -f "$sibling/evaluation_log.csv" ]; then
                CSV_LOG="$sibling/evaluation_log.csv"
                break
            fi
        done
    fi
    if [ -f "$CSV_LOG" ]; then
        LINES=$(wc -l < "$CSV_LOG")
        echo "CSV Log: $((LINES - 1)) records"

        # Quick stats using python
        python3 -c "
import pandas as pd
df = pd.read_csv('$CSV_LOG')
if len(df) > 0:
    print('')
    # Separate probing from experiment data
    probing_df = df[df['policy'] == '_probing']
    exp_df = df[df['policy'] != '_probing']

    if len(probing_df) > 0:
        probing_cost = probing_df['cost_usd'].sum()
        print(f'Probing: n={len(probing_df)}, cost=\${probing_cost:.4f}')

    print('')
    print('Policy Statistics:')
    for policy in sorted(exp_df['policy'].unique()):
        pdf = exp_df[exp_df['policy'] == policy]
        success = pdf[pdf['status'] == 'success']
        if len(success) > 0:
            p50 = success['ttft_ms'].median()
            p99 = success['ttft_ms'].quantile(0.99)
            slo_viol = pdf['slo_violated'].mean() * 100
            cost = pdf['cost_usd'].sum()
            err_rate = (1 - len(success) / len(pdf)) * 100
            print(f'  {policy}: n={len(pdf)}, P50={p50:.0f}ms, P99={p99:.0f}ms, SLO={slo_viol:.1f}%, err={err_rate:.1f}%, cost=\${cost:.4f}')
        else:
            print(f'  {policy}: n={len(pdf)}, no successful requests')
" 2>/dev/null || echo "  (unable to read CSV)"
    fi
    echo ""
    echo "=============================================="
}

# Tail log output
tail_log() {
    RUN_DIR="${1:-$(find_latest_run)}"

    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "No experiment found"
        exit 1
    fi

    LOG_FILE="$RUN_DIR/nohup.log"
    if [ -f "$LOG_FILE" ]; then
        echo "Following: $LOG_FILE (Ctrl+C to stop)"
        tail -f "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
        exit 1
    fi
}

# Stop experiment
stop_experiment() {
    RUN_DIR="${1:-$(find_latest_run)}"

    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "No experiment found"
        exit 1
    fi

    PID_FILE="$RUN_DIR/pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping experiment (PID: $PID)..."
            kill "$PID"
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Force killing..."
                kill -9 "$PID"
            fi
            echo "Stopped."
        else
            echo "Experiment not running"
        fi
    else
        echo "No PID file found"
    fi
}

# Main
case "${1:-start}" in
    start)
        start_experiment
        ;;
    resume)
        resume_experiment "$2"
        ;;
    status)
        check_status "$2"
        ;;
    tail)
        tail_log "$2"
        ;;
    stop)
        stop_experiment "$2"
        ;;
    *)
        echo "Usage: $0 {start|resume|status|tail|stop} [run_directory]"
        echo ""
        echo "Commands:"
        echo "  start   - Start new 24h experiment"
        echo "  resume  - Resume from checkpoint"
        echo "  status  - Check experiment progress"
        echo "  tail    - Follow live output"
        echo "  stop    - Stop running experiment"
        exit 1
        ;;
esac
