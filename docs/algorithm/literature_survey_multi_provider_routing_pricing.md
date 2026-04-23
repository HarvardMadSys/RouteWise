# Literature Survey: Multi-Provider LLM API Routing and Pricing

> Prepared for: Juncheng's request — "can you check whether anyone has worked on multi-provider routing and pricing?"

---

## Key Finding

**Same-model multi-provider routing (choosing between providers for the SAME model) is almost entirely unexplored in academia.** Existing work focuses on cross-model routing (choosing between different models). This is a clear gap and a strong positioning for our work.

However, multiple related threads exist: (1) cross-model LLM routing, (2) cloud capacity planning / stochastic programming, (3) market pricing / mechanism design, (4) LLM inference economics, and (5) speculative execution for agentic workloads.

---

## 1. LLM Routing (Cross-Model)

These papers route queries across DIFFERENT models for cost-quality tradeoff. None addresses same-model multi-provider routing.

### 1.1 Foundational Works

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **FrugalGPT** | Chen, Zaharia, Zou | ICLR 2025 | LLM cascade: query cheap models first, escalate if unreliable. 98% cost reduction matching GPT-4. Three strategies: prompt adaptation, LLM approximation, LLM cascade. |
| **RouteLLM** | Ong et al. (LMSYS/Berkeley) | ICLR 2025 | Train router on human preference data to route between strong/weak models. 2x+ cost reduction. Open-source framework with 4 pre-trained routers. |
| **Hybrid LLM** | Ding et al. (Microsoft) | ICLR 2024 | Quality-aware router between large (cloud) and small (edge) models. Reduces 40% large model calls with uncertainty-aware routing. |
| **AutoMix** | Aggarwal, Madaan et al. | NeurIPS 2024 | POMDP-based router with few-shot self-verification. No training data needed. 50%+ compute reduction. |

### 1.2 Unified Routing + Cascading

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Unified Routing & Cascading** | Dekoninck et al. (ETH) | ICLR 2025 | First to unify routing and cascading with optimality proofs. Cascade routing can skip/reorder models. +4% over either alone on RouterBench. |
| **SATER** | (Various) | EMNLP 2025 | Self-aware token-efficient routing/cascading. 50%+ cost and 80%+ cascade latency reduction. |
| **C3PO** | Valkanas et al. | NeurIPS 2025 | Self-supervised cascade with conformal prediction for probabilistic cost constraints. Formal cost control guarantees. |
| **Cascadia** | Jiang et al. | arXiv 2506 | Joint optimization of routing + deployment via bi-level optimization (MILP + Chebyshev). |

### 1.3 Online Learning Approaches

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **MetaLLM** | (Various) | arXiv 2407 | Multi-armed bandit framework for dynamic model routing. Lightweight, works with any LLM pool. |
| **BaRP** | (Various) | arXiv 2510 | Preference-conditioned contextual bandit. Adjustable cost/latency/quality tradeoff at inference time without retraining. |
| **PILOT** | Panda et al. | EMNLP 2025 Findings | LinUCB-based routing with budget as multi-choice knapsack constraint. 93% of GPT-4 at 25% cost. |
| **BEST-Route** | (Microsoft Research) | ICML 2025 | Combines routing with best-of-n sampling. Small model x N samples can beat large model x 1. 60% cost reduction. |
| **Online Multi-LLM Selection** | (Various) | arXiv 2506 | Contextual bandit with sublinear regret, no future context prediction needed. |

### 1.4 Advanced Router Architectures

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **EmbedLLM** | Zhuang et al. | ICLR 2025 Spotlight | Learn compact LLM embeddings for routing. One representation for multiple downstream tasks. |
| **ZeroRouter** | Yan et al. | arXiv 2601 | Universal latent space for zero-shot routing. New models onboarded without retraining. Solves cold-start. |
| **Router-R1** | (UIUC) | arXiv 2506 | RL-trained LLM as router. Multi-round routing with cost reward. Generalizes to new models. |
| **Arch-Router** | (Katanemo) | arXiv 2506 | 1.5B compact router. Natural language routing policies. New models without retraining. |
| **DiSRouter** | (Various) | arXiv 2510 | Distributed self-routing. Each agent decides locally. Scales better than centralized routing. |

### 1.5 Routing with Cost + Latency Constraints

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **SCORE** | Shahout, Yu et al. (Harvard) | Workshop 2025 | **Jointly optimizes cost AND latency** (most work ignores latency). Uses quality + length predictors. |
| **QC-Opt** | Shekhar et al. | arXiv 2402 | Budget + latency constrained optimization. 40-90% cost reduction, 4-7% quality improvement. |
| **OptiRoute** | Piskala et al. | arXiv 2502 | Multi-dimensional user preferences (cost, latency, ethics). kNN + hierarchical filtering. |

### 1.6 Critical Analysis of Routing

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **kNN Beats Learned Routers** | Li et al. | arXiv 2505 | Simple kNN matches/beats complex learned routers. Standardized benchmarks. |
| **EquiRouter (Routing Collapse)** | Lai, Ye | arXiv 2602 | Routers degenerate to always picking the strongest model as budget increases. Objective-decision mismatch. |
| **Rerouting LLM Routers** | Shafran et al. (Hebrew U) | COLM 2025 | Adversarial attacks on routing. "Confounder gadgets" force routing to strong model. |

### 1.7 Benchmarks

| Benchmark | Venue | Scale | Key Finding |
|-----------|-------|-------|-------------|
| **RouterBench** (Martian) | arXiv 2403 | 405K+ results | First routing benchmark |
| **RouterArena** | arXiv 2510 | Open platform | Comprehensive router comparison |
| **LLMRouterBench** | arXiv 2601 | 400K instances, 33 models | **Many methods don't beat simple baselines** |

### 1.8 Pricing Game Theory for LLM Routing

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **PriLLM** | Guo et al. | arXiv 2511 | **Stackelberg game for LLM routing market**. Provider sets price, user routes based on cost/QoS. Deep aggregation network for scalable dynamic pricing. |

---

## 2. Cloud Capacity Planning and Stochastic Programming

These papers address the **reserved vs on-demand** decision under demand uncertainty - directly relevant to Juncheng's suggestion about capacity blocks.

### 2.1 Multi-Stage Stochastic Programming

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Cloud Resource Provisioning via SP** | Nair et al. | EJOR 2020 | Multi-stage SP for IaaS. Reserved (cheap, committed) vs on-demand (expensive, flexible). Chance constraints for service levels. AWS EC2 validation. |
| **OCRP Algorithm** | Chaisiri, Lee, Niyato | IEEE TSC 2012 (641 cit.) | Optimal Cloud Resource Provisioning. Benders decomposition + sample-average approximation. **49% cost reduction**. |
| **Value of Multistage SP** | Huang, Ahmed | Operations Research 2009 | Proves multistage SP is significantly better than two-stage for capacity planning. Efficient approximation scheme with asymptotic optimality proof. |
| **Cloud Cost Optimization** | Qu, Dawande, Janakiraman | Operations Research 2024 | Infinite-horizon stochastic optimization. Decoupling method for lower bounds. Asymptotically optimal (gap shrinks as O(1/sqrt(theta))). AWS pricing validation. |

### 2.2 Demand Surge and Newsvendor Models

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Capacity Reservation for Demand Surges** | Chen, Lei, Moinzadeh | POM 2024 | Newsvendor-type optimal capacity formula. Reserved for base demand + short contracts for surges. **Analyzes value of secondary market for capacity trading.** |
| **Nonstationary Newsvendor** | (Various) | arXiv 2305 | Sequential newsvendor under non-stationary demand. Online algorithms with/without predictions. |
| **Optimal Capacity Planning** | Furman, Diamant | EJOR 2025 | Admission control with periodic time-varying demand. 24-hour cloud request data validation. |

### 2.3 Supply-Side Capacity Decisions

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Cloud Server Deployment** | Liu et al. (Microsoft Research) | M&SOM 2025 | Two-stage SP for server deployment. Risk-aware cutting-plane method. |
| **Capacity Expansion with Bundled Resources** | Arbabian, Chen, Moinzadeh | M&SOM 2021 | Capacity expansion when attributes (CPU, RAM, GPU) come bundled but demand is asymmetric. |
| **Cloud Capacity Planning Survey** | Shi Chen | FnT TOM 2024 | Comprehensive survey. Joint treatment of time uncertainty + demand uncertainty. |

---

## 3. Market Design, Pricing Mechanisms, and Spot Markets

Directly relevant to Juncheng's vision of "capacity block" and "win-win for OpenRouter and API provider".

### 3.1 Cloud Pricing Theory

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Cloud Pricing: Spot Market Strikes Back** | Dierks, Seuken | Management Science 2022 | Game theory + queueing theory. Conditions for when spot + on-demand is profitable. **Market cannibalization** analysis. |
| **Selling Cloud to Risk-Averse Customers** | Dierks, Seuken | WINE 2017 | Mechanism design for risk-averse cloud users choosing between on-demand and spot. |
| **Fixed and Market Pricing** | Abhishek, Kash, Key | NetEcon 2012 | Linked queueing model. Fixed price usually generates higher revenue than hybrid market. |
| **Deconstructing EC2 Spot Pricing** | Ben-Yehuda et al. | ACM TEC 2013 | Reverse-engineers AWS spot pricing. 98% of prices from artificial algorithm, not true supply-demand. |

### 3.2 Two-Sided Market Design

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Two-Sided Market for Computing** | Mashayekhy et al. | IEEE BigData 2014 | Strategy-proof mechanism for trading big data compute. Buyers and sellers reveal true values. |
| **Dynamic Combinatorial Double Auction** | (Various) | J. Cloud Computing 2023 | Truthful dynamic combinatorial double auction. Greedy mechanism satisfying IC, IR, budget balance. |
| **Cloud Computing Value Chains** | Chen, Moinzadeh, Song, Zhong | M&SOM 2023 | OM perspective survey: resource management, market pricing, capacity supply chain. |

### 3.3 GPU Compute Markets

| Initiative | Who | Year | Key Idea |
|-----------|-----|------|----------|
| **Auctionomics x OneChronos** | Paul Milgrom (Nobel 2020) | 2025 | **First tradable financial market for GPU compute**. Combinatorial auctions for GPU bundles (type, duration, location). Bilateral forwards for price discovery. |

### 3.4 Revenue Management and Dynamic Pricing

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Dynamic Cloud Pricing** | Xu, Li | IEEE TCC 2013 | Stochastic dynamic programming for revenue maximization under random demand. |
| **Real-Time Dynamic Pricing** | Lei, Jasin | Operations Research 2020 | Reusable resources + advance reservation + deterministic service time. Asymptotically near-optimal. |
| **SLA Trifecta** | Yuan et al. | ISR 2018 | Joint optimization of backup resources, price, and penalty in availability-aware cloud. |

---

## 4. Online Algorithms: Ski-Rental and Buy-or-Rent

Directly relevant to our primal-dual framework and learning-augmented approach.

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Two-Level Ski-Rental** | (Various) | **AAAI 2024** | Three options: rent (on-demand), single purchase (per-item reservation), combo (full reservation). **Learning-augmented** algorithm (LADTSR) beats robust online with good predictions, maintains worst-case guarantees with bad predictions. |
| **Multi-Option Ski Rental** | (Various) | arXiv 2302 | Best-possible competitive analysis for multi-option ski rental with ML predictions. |
| **Constrained Ski-Rental** | Khanafer et al. | IEEE INFOCOM 2013 | Ski-rental with known moments of demand distribution. Cloud cost optimization application. |
| **Rent, Lease, or Buy** | Lotker et al. | SIAM J. DM 2012 | Multi-slope ski-rental (multiple options with different fixed + variable costs). Optimal randomized strategy is e-competitive. |
| **OOLR** | Monteil et al. | IEEE 2023 | Optimistic Online Learning for Reservation. FTRL-based with prediction integration. O(sqrt(T)) regret. |
| **Bahncard Problem** | (Various) | arXiv 2410 | Learning-augmented algorithms for the Bahncard problem (generalized ski-rental). |
| **LLM Fine-Tuning on Spot** | (Various) | arXiv 2512 | Deadline-aware scheduling on spot + on-demand mix. Committed horizon control. |

---

## 5. LLM Inference Economics and Market Analysis

### 5.1 Market Structure Studies

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **The Emerging Market for Intelligence** | Demirer, Fradkin, Tadelis, Peng | **NBER Working Paper 2025** | **First comprehensive study of LLM market structure** using OpenRouter + Azure data. Documents: rapid growth, price declines with persistent heterogeneity, open-source 90% cheaper than closed-source, price elasticities just above 1. |
| **Demand for LLMs** | Fradkin | arXiv 2504 | Three facts from OpenRouter data: rapid adoption of new models, substitution vs market expansion, widespread multihoming. |
| **Economics of LLMs** | Bergemann (Yale), Bonatti (MIT), Smolin | **ACM EC 2025** | Formal pricing theory. Optimal pricing uses menus of two-part tariffs. Rationalizes industry tiered pricing. |
| **Inference Economics** | Erdil (Epoch AI) | arXiv 2506 | Cost-speed tradeoff theory. Pareto frontiers of speed vs cost-per-token. Explains why providers offer different price/latency points. |
| **State of AI (100T Tokens)** | Aubakirova et al. (a16z + OpenRouter) | arXiv 2601 | 100T token empirical study. Programming grew to 50%+ of volume. Open-source ~33%. **Agentic inference is fastest growing**. Providers grew from 27 to 90 in 2025. |

### 5.2 Cost Trends

| Paper/Report | Source | Key Finding |
|-------------|--------|-------------|
| **LLMflation** | a16z (Appenzeller) 2024 | 10x/year cost decline. GPT-3 level: $60/M tokens (2021) -> $0.06 (2024). 1000x in 3 years. |
| **Price of Progress** | arXiv 2511.23455 | Algorithmic efficiency falling at 5-10x/year. Uneven across task categories (9x-900x/yr). |
| **On-Premise Break-Even** | Pan et al., arXiv 2509 | Framework for when self-hosting beats API. |

### 5.3 Benchmarking Platforms

| Platform | What it does |
|----------|-------------|
| **Artificial Analysis** | Real-time measurement of 500+ endpoints. TTFT, output speed, blended price. |
| **Intelligence Per Watt** (Stanford) | IPW metric for local inference efficiency. 5.3x improvement 2023-2025. |
| **AI Ping** (Tsinghua/QingCheng) | 30+ Chinese API providers. 7x24 monitoring. Smart routing based on real-time profiling. |
| **TokenPowerBench** | arXiv 2512 | Tokens-per-Watt benchmarks across hardware. |

---

## 6. Speculative Execution for Agentic Workloads

Directly relevant to Juncheng's "speculative tool call" idea.

### 6.1 Speculative Agent Actions

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **Speculative Actions** | Ye et al. | arXiv 2510 | **Predict likely next actions using faster models, execute in parallel.** 55% prediction accuracy. Lossless speedup. Gaming, e-commerce, web search environments. |
| **Speculative Tool Calls** | Nichols et al. | arXiv 2512 | **Systems-level: tool proposer in vLLM's speculative decoding infrastructure.** Detects tool-call boundaries, drafts from cache. Proposes "tool cache" API endpoint. |
| **Dynamic Speculative Planning** | Guan et al. | arXiv 2509 | Async RL for speculation in multi-step agent decisions. Dual-agent architecture. |
| **SPAgent** | Huang et al. (Tsinghua) | arXiv 2511 | Two-phase adaptive speculation with load-aware scheduling. 1.65x e2e speedup. 23.8% LLM inference time reduction. |
| **Sherlock** | Ro et al. (Microsoft) | arXiv 2511 | Speculative downstream execution + selective verification with rollback. 48.7% latency reduction, 26% cost reduction. |

### 6.2 Speculative Decoding (Provider-Side)

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **SuffixDecoding** | Oliaro et al. (CMU) | NeurIPS 2025 | Model-free spec decoding via suffix trees. 2.8x faster than EAGLE on agentic SQL. No draft model needed. |
| **Mirror Speculative Decoding** | Bhendawade et al. (Apple) | arXiv 2510 | Bidirectional speculation on GPU+NPU. 2.8x wall-time speedup. |

### 6.3 Agentic Workload Analysis

| Paper | Authors | Venue | Key Idea |
|-------|---------|-------|----------|
| **What Limits Agentic Efficiency?** | Bian et al. (UW-Madison) | arXiv 2510 | **Larger models can sometimes have lower latency than smaller ones** due to API variability. Challenges naive size-based routing. |
| **AgentCgroup** | (Various) | arXiv 2602 | **OS-level execution = 56-74% of e2e latency. LLM reasoning = only 26-44%.** Tool calls dominate. |
| **LLMCompiler** | Kim et al. | ICML 2024 | Parallel function calling planner. Reduces sequential LLM calls. |
| **W&D** | Lin et al. | arXiv 2602 | Width (parallel tool calls) + Depth scaling for research agents. |
| **Plan Caching** | Zhang et al. | NeurIPS 2025 | Cache reusable agent plans. 50% cost reduction, 27% latency reduction. |

---

## 7. Industry Solutions (LLM API Gateways)

| Platform | Routing Strategy | Notes |
|----------|-----------------|-------|
| **OpenRouter** | Stability-first + inverse-square-of-price weighting. Auto-fallback. | **Our primary comparison target.** Simple heuristics, no theoretical guarantees. |
| **Unify.ai** | Live benchmarks (10-min refresh). Quality + cost + speed routing. Custom constraints. | Most similar to our latency profiling approach. |
| **LiteLLM** | Simple-shuffle (default), latency-based, cost-based routing. | Open-source baseline. 100+ providers. |
| **Portkey** | Fallback + load balancing + conditional routing. 200+ models/250+ providers. | Reliability-focused, not optimization-focused. |
| **Martian** | Model Mapping via mechanistic interpretability. | Cross-model routing. 300+ companies. |
| **Not Diamond** | Learned router for model selection. | Maintains awesome-ai-model-routing list. |

---

## 8. Summary: Research Gaps and Our Positioning

### Gap 1: Same-Model Multi-Provider Routing
- **All existing LLM routing papers** focus on cross-model routing (e.g., GPT-4 vs GPT-3.5).
- **No academic work** addresses choosing between providers for the same model.
- Industry (OpenRouter, Unify) does this with simple heuristics.
- **Our work fills this gap** with principled algorithms and theoretical guarantees.

### Gap 2: Capacity Block Pricing for LLM APIs
- Cloud capacity planning is mature (stochastic programming, newsvendor models).
- **No one has applied these frameworks to LLM API markets.**
- The two-level ski-rental (AAAI 2024) is the closest theoretical analog.
- **Opportunity**: Formulate OpenRouter's provider negotiation as capacity block procurement.

### Gap 3: Joint Cost-Latency-Quality Routing with Market Design
- Most routing papers optimize cost-quality only. SCORE (Harvard) adds latency.
- PriLLM adds pricing game theory but not capacity planning.
- **No one combines**: routing algorithms + capacity planning + win-win market design.
- **Our opportunity**: Show that capacity blocks + intelligent routing = win-win.

### Gap 4: Routing for Agentic Workloads
- AgentCgroup shows tool calls dominate latency (56-74%).
- Speculative tool calling papers exist but don't consider multi-provider routing.
- **Opportunity**: Routing optimization that accounts for tool-call-dominated latency structure.

### Key Papers to Cite in Each Section

**Related Work (Routing)**:
- FrugalGPT (ICLR 2025), RouteLLM (ICLR 2025), Hybrid LLM (ICLR 2024)
- Unified Routing & Cascading (ICLR 2025), C3PO (NeurIPS 2025)
- LLMRouterBench (2026)

**Related Work (Capacity/Pricing)**:
- Cloud Cost Optimization (OR 2024), Two-Level Ski-Rental (AAAI 2024)
- Dierks & Seuken (Management Science 2022)
- Cloud Computing Value Chains (M&SOM 2023)

**Related Work (Market)**:
- Emerging Market for Intelligence (NBER 2025)
- Economics of LLMs (ACM EC 2025)
- State of AI (a16z + OpenRouter 2026)

**Related Work (Agentic/Speculative)**:
- Speculative Actions (arXiv 2510), Speculative Tool Calls (arXiv 2512)
- AgentCgroup (arXiv 2602), What Limits Agentic Efficiency (arXiv 2510)
