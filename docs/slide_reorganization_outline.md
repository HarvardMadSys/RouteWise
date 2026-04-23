# Slide Reorganization Outline

## Juncheng's Feedback
1. Don't introduce new concepts in Evaluation - all design should be explained before
2. Cost and Latency should both appear in Design, not just cost
3. End-to-end experiment (OpenRouter 24h) should come FIRST in Evaluation
4. Follow standard systems paper structure (NSDI/OSDI style)

---

## Current Structure (47 pages, problems marked with >>)

```
1. Title + Outline
2. Motivation & Problem (3 frames)
   - Cost Challenge
   - Latency Challenge
   - Research Questions
3. Offline Algorithms (3 frames)
   - Quota routing
   - Joint routing
   - ILP formulation
4. Online Algorithms (6 frames)
   - Challenges
   - Greedy baseline
   - Primal-Dual
   - Threshold visualization
   - Worked example
   - Unknown output length (EMA/Histogram)
   - Learning-Augmented (LA-PD)
5. Experiments (9 frames)                    >> cost only, no latency design yet
   - Setup
   - Offline quota results
   - Sensitivity analysis
   - Online cost comparison
   - Ablation study
   - Predictor calibration
   - Oracle sensitivity
   - Hyperparameter sensitivity
   - Stage 2 joint optimization
   - Single-model comparison
6. Latency-Aware Routing (8 frames)          >> design + evaluation mixed together
   - Phase 1: Profiling (design)
   - Phase 2: LP-Mix (design)
   - Phase 2 Results (evaluation)
   - Phase 3: Online routing (design)
   - Phase 3 Results (evaluation)
   - Phase 4: Smart hedging (design)
   - Phase 4 Results (evaluation)
   - OpenRouter 24h (end-to-end)             >> best experiment buried at the end!
   - Latency summary
7. Contributions & Future Work (4 frames)
   - Implementation & Scalability
   - Summary                                 >> contains buggy Stage 2 numbers
   - Discussion: subscription complexity
   - Limitations & Future Work
```

---

## Proposed New Structure

```
1. TITLE + OUTLINE (2 frames)

2. MOTIVATION & PROBLEM (3 frames)
   [Keep as-is, already covers both cost and latency]
   - Cost Challenge: Hybrid Pricing & Opportunity Cost
   - Latency Challenge: Provider Heterogeneity & Drift
   - Research Questions

3. SYSTEM DESIGN: COST OPTIMIZATION (7 frames)
   [Merge offline + online into one "design" section]
   3.1 Offline: Quota-Constrained Routing (S_Q + S_A)
   3.2 Offline: Joint Routing ILP (S_Q + S_C + S_A)
   3.3 Online: Challenges & Greedy Baseline
   3.4 Online: Primal-Dual Threshold Strategy
       (merge current "threshold viz" into this frame)
   3.5 Online: Handling Unknown Output Length
       (EMA predictor - explain here so evaluation just shows numbers)
   3.6 Online: Learning-Augmented Variant (LA-PD)
   3.7 Worked Example: Primal-Dual vs Greedy

4. SYSTEM DESIGN: LATENCY OPTIMIZATION (4 frames)
   [Move design parts here, leave results for evaluation]
   4.1 Provider Profiling & CDF Estimation
   4.2 LP-Based Optimal Provider Mixing
   4.3 Online Adaptive Routing (SWRR + drift detection)
   4.4 Smart Hedging: Cost-Benefit Rule
       (explain the formula here, not in evaluation)

5. IMPLEMENTATION (1 frame)
   [Brief: routing decision <50us, ILP solver perf, code stats]
   - Move current "Implementation & Scalability" frame here

6. EVALUATION (12-13 frames)
   [All concepts already introduced - just show results]

   6.1 Experimental Setup (1 frame)
       - Datasets, pricing, baselines

   6.2 End-to-End: 24h OpenRouter Production (1 frame)  *** FIRST ***
       - The "wow" result: 31x fewer violations, 37% cost savings
       - Shows both cost AND latency together

   6.3 Cost Optimization Deep-Dive (6-7 frames)
       - Offline: Quota routing results + sensitivity
       - Online: PD-EMA cost comparison (BurstGPT)
       - Ablation: Predictor vs Decision Rule
       - Oracle Sensitivity: Do we need better prediction?
       - Stage 2: Joint Optimization (corrected results)
       - Single-Model Comparison: S_C value analysis

   6.4 Latency Optimization Deep-Dive (3-4 frames)
       - LP-Mix results: optimal routing decisions
       - Online routing: adaptive performance
       - Hedging comparison: strategy ablation
       - (Pareto frontier + hedge rate plots)

7. DISCUSSION & CONCLUSION (3 frames)
   - Summary of Contributions (with corrected numbers)
   - Discussion: Real-World Subscription Complexity
   - Limitations & Future Work

APPENDIX (keep as-is)
```

---

## Key Changes Summary

| Change | Why |
|--------|-----|
| Latency design moved before Evaluation | No new concepts in evaluation |
| OpenRouter 24h moved to Eval #1 | Best result first (Juncheng's suggestion) |
| Offline + Online merged into "Cost Design" | Cleaner narrative arc |
| Latency "Phase 1-4" restructured | Separate design from results |
| Implementation as standalone section | Standard NSDI structure |
| Removed: Threshold visualization frame | Content merged into PD frame |
| Removed: Predictor calibration frame | Can be appendix |
| Removed: Latency summary frame | Redundant with conclusion |

## Frames to DELETE or MOVE to Appendix
- "Threshold Visualization" (501) -> merge key diagram into PD frame
- "Predictor Calibration Analysis" (951) -> appendix (too detailed for main)
- "Latency Routing: Summary" (1531) -> delete (redundant)
- "Hyperparameter Sensitivity" (1049) -> could go to appendix

## Estimated Page Count
- Current: 47 pages (too many for a talk)
- Target: ~35-38 pages (more focused)

## What Needs Rewriting
1. Section headers (\section{}) need renaming
2. Latency frames need splitting (design part vs result part)
3. "Summary of Contributions" frame needs corrected Stage 2 numbers
4. Outline frame auto-updates from \section{}
