# Agentic Workload Routing & Speculative Tool Call

> Juncheng's idea analysis + literature connections + action plan

## 1. The Three-Layer Idea

### Layer 1: LLM Is Not Always the Bottleneck

**Observation:** In agentic workloads, tool calls (web search, code execution, DB query) often dominate end-to-end latency, not LLM inference.

```
Agentic workflow per step:
  LLM generate (1-5s) → Tool execute (5-30s) → Result → LLM generate → ...
```

**Evidence:**
- Bian et al. "What Limits Agentic Systems Efficiency?" (arXiv 2510.16276): tool calls account for 56-74% of agentic latency
- Parrot (OSDI 2024): multi-request LLM applications have complex inter-request dependencies with I/O-bound tool calls
- Pie (SOSP 2025): agentic workflows see 1.3-3.4x improvement when I/O is interleaved with generation

**Implication:** Optimizing LLM inference latency has diminishing marginal returns for agentic workloads.

### Layer 2: Choose Slower but Cheaper Provider (Fits Current Paper)

**Idea:** If `tool_call_latency >> LLM_latency`, the effective latency SLO on the LLM is relaxed. We can route more aggressively to cheaper (but slower) providers without affecting user-perceived end-to-end latency.

**In our paper's language:**
- Agentic workload's effective latency budget allocates less to LLM
- On the cost-latency Pareto frontier, we can pick a more cost-optimal operating point
- This is a workload-aware latency budget in our LP-based provider mixing:
  - **Chat query:** latency budget fully allocated to LLM → need fast provider
  - **Agentic query:** latency budget mostly for tool call → LLM can use slow/cheap provider

**How to add to current paper:**
1. Collect agentic workload traces (SWE-bench agent traces, WebArena, or similar)
2. Analyze tool call latency vs LLM latency ratio
3. Show that in agentic setting, our routing algorithm saves more money (relaxed latency constraint)
4. Compare cost savings: agentic vs non-agentic workloads

**Effort:** ~0.5-1 page of content + 1 experiment figure. Lightweight extension, does not break the "same-model cross-provider" narrative.

### Layer 3: Speculative Tool Call (Separate Paper)

**Idea:** While the large model is still decoding, use a small/fast model to predict the tool call and pre-execute it. When the large model confirms, the result is already ready.

```
Timeline (no speculation):
[Large model decode 3s] → [Get tool call] → [Execute tool 5s] → Total 8s

Timeline (with speculation):
[Large model decode 3s           ]
[Small model 0.5s → Execute tool 5s]  ← parallel
→ Correct prediction: Total 5.5s (saved 2.5s)
→ Wrong prediction:   Total 8s (same as no speculation)
```

**Analogy to existing CS concepts:**

| Concept | Draft | Verify | Misprediction Cost |
|---------|-------|--------|--------------------|
| CPU speculative execution | Predict branch, execute ahead | Check branch condition | Pipeline flush (rollback) |
| Speculative decoding | Small model drafts tokens | Large model verifies in parallel | Discard wrong tokens (no quality loss) |
| **Speculative tool call** | Small model predicts tool call, pre-execute | Large model confirms actual tool call | Wasted tool execution (+ possible side effects) |

## 2. Building Blocks from Existing Literature

### Direct Mappings

#### 2.1 Speculative Decoding → Speculative Tool Call

**Online Speculative Decoding** (Liu et al., ICML 2024):
- Core insight: draft model adapts online to user distribution, improving over time
- Mapping: predictor doesn't need to be accurate from the start, can learn on the fly
- For agentic workloads: same agent's tool call patterns are highly repetitive
  - Coding agent: repeatedly `read_file` → `edit_file` → `run_test`
  - Online adaptation should work even better than token-level speculation

#### 2.2 FrugalGPT Cascade → Tool Call Cascade

**FrugalGPT** (Chen et al., TMLR 2024) three components: router → scorer → stop judger

Mapping to speculative tool call:
1. Small model generates tool call prediction
2. **Scorer** evaluates prediction confidence (is this tool call prediction reliable?)
3. **Stop judger** decides: high confidence → pre-execute; low confidence → wait for large model

This avoids the wasted computation problem of blind speculation.

#### 2.3 Router-R1 RL Framework → Learn When to Speculate

**Router-R1** (Zhang et al., 2025) uses RL for sequential routing decisions.

Mapping: train a **speculation policy** with RL:
- **State:** current conversation context, tool call history, agent type
- **Action:** speculate (predict + pre-execute) or wait (let large model finish)
- **Reward:** `latency_saved - cost_of_misprediction`

Router-R1's cost reward coefficient directly applies: balance speculation benefit vs misprediction cost.

### Framework-Level Connections

#### 2.4 Parrot's Semantic Variable → Expose DAG Structure

**Parrot** (OSDI 2024): Semantic Variable abstraction exposes inter-request dependencies in agentic workflows as a DAG.

Value for speculative tool call: if you know the DAG structure, you can identify **which tool calls are predictable**:
- Coding agent: `generate_code` → 90% followed by `run_test` → worth speculating
- Search agent: `search(query)` → query content is uncertain → not worth speculating

#### 2.5 Pie's Inferlets → Execution Model

**Pie** (SOSP 2025): allows interleaving I/O during generation via "inferlets" (WebAssembly programs).

This is almost exactly the execution infrastructure needed for speculative tool calls — execute tool call I/O in parallel with main generation. Pie provides the systems substrate.

#### 2.6 Our Smart Hedging → Speculation Hedging

Our Phase 4 hedging logic extends naturally:

| Our Paper's Hedging | Speculative Tool Call Hedging |
|---|---|
| Primary provider latency uncertain → hedge to backup | Tool call prediction uncertain → pre-execute predicted + wait for actual |
| Survival analysis decides when to hedge | Confidence score decides when to speculate |
| Misprediction cost = extra money spent | Misprediction cost = wasted tool execution |

## 3. Paper Skeleton: Speculative Tool Execution

If developed as a standalone paper:

```
Title: Speculative Tool Execution for Agentic LLM Workloads

1. Motivation
   - Tool calls dominate agentic latency (data from Parrot, Pie, Bian et al.)
   - Opportunity: overlap tool execution with LLM generation

2. Problem Formulation
   - Given: agentic workflow with LLM calls and tool calls
   - Decision: for each LLM generation step, speculate or wait?
   - Objective: minimize end-to-end latency subject to speculation budget (wasted compute)

3. System Design
   - Draft-verify architecture (from speculative decoding)
   - Confidence-based cascade for speculation gating (from FrugalGPT)
   - DAG-aware selective speculation (from Parrot)
   - Interleaved I/O execution (from Pie)

4. Speculation Policy
   - Option A: RL-based policy (from Router-R1)
   - Option B: Threshold-based with online adaptation (from Online Spec Decoding)
   - Hedging mechanism for uncertain predictions (from our smart hedging)

5. Safety
   - Read-only tools: safe to speculate (search, read_file, etc.)
   - Write tools: need confirmation before execution (edit_file, send_email)
   - Side-effect classification as part of speculation policy

6. Evaluation
   - Benchmarks: SWE-bench, WebArena, ToolBench
   - Metrics: end-to-end latency, speculation accuracy, wasted computation
   - Baselines: no speculation, always speculate, oracle speculation
```

## 4. Recommendations

### For Current Paper (NSDI submission)
- **Add Layer 2** as an experiment/discussion section (~0.5-1 page)
- Show workload-aware routing saves more for agentic workloads
- Mention Layer 3 briefly in conclusion as future work

### For Next Project
- Layer 3 (speculative tool call) as standalone paper
- Target venue: OSDI / SOSP / NSDI (systems) or ICML / NeurIPS (if more algorithmic)
- Building blocks from literature are clear; technical path is well-defined
- Key open question: prediction accuracy of small model for tool calls (needs empirical study)

### Key Risk for Layer 3
- Prediction accuracy: how well can a small model predict large model's tool call?
- Side effects: speculating write operations is dangerous
- Scope: might need to limit to read-only tool speculation initially
