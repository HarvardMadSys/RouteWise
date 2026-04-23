# Literature Survey: Multi-Provider Routing & Pricing (Top Venues Only)

> Filtered to: OSDI, SOSP, NSDI, EuroSys, MLSys, ASPLOS, ISCA, FAST, ICLR, ICML, NeurIPS, Management Science, Operations Research, M&SOM, ACM EC, NBER
> Labs: Berkeley Sky/LMSYS, Stanford, CMU, UW-Madison, MIT Han Lab, ETH Zurich, Microsoft Research

---

## Key Finding

**Same-model multi-provider routing has zero academic publications at top venues.** All existing LLM routing work (ICLR/ICML/NeurIPS) routes across *different models*. The systems community (OSDI/SOSP/EuroSys) builds serving infrastructure but does not optimize *cross-provider* routing. The OR community (Management Science/Operations Research) studies cloud pricing but has not applied it to LLM APIs. **This is our gap.**

---

## A. LLM Routing & Cascading (Direct Competitors)

| Paper | Venue | Lab | Key Idea | Gap vs. Our Work |
|-------|-------|-----|----------|-------------------|
| **RouteLLM** (Ong, Stoica et al.) | **ICLR 2025** | Berkeley LMSYS | Train router on preference data to select strong/weak model. 2x+ cost reduction. | Cross-model only. No latency, no provider selection. |
| **Hybrid LLM** (Ding et al.) | **ICLR 2024** | Microsoft | Quality-aware router between large/small models. 40% fewer large model calls. | Two-model only. No multi-provider. |
| **Unified Routing & Cascading** (Dekoninck et al.) | **ICML 2025** | ETH Zurich | Provably optimal cascade routing framework. +14% on RouterBench. | Theory for cross-model. No provider/latency dimension. |
| **AutoMix** (Aggarwal, Madaan et al.) | **NeurIPS 2024** | CMU/IIT Delhi | POMDP-based router with self-verification. 50%+ cost reduction. <1ms overhead. | Cross-model. No capacity/pricing considerations. |
| **FrugalGPT** (Chen, Zaharia, Zou) | **TMLR 2024** | Stanford | LLM cascade: query cheap first, escalate if unreliable. 98% cost reduction. | Foundational but cross-model cascade only. |
| **RouterDC** (Chen et al.) | **NeurIPS 2024** | HKUST | Dual contrastive learning router. +2.76% over best single model. | Cross-model selection. |
| **Archon** (Saad-Falcon, Re, Mirhoseini) | **ICML 2025** | Stanford | Architecture search over LLM ensembles. +15.1% over GPT-4o/Claude 3.5. | Multi-model composition, not provider routing. |
| **Compound AI Scaling** (Chen, Stoica, Zaharia, Zou) | **NeurIPS 2024** | Stanford/Berkeley | Scaling laws for multi-LLM-call systems. Non-monotonic: more calls can hurt. | Informs budget allocation but not provider selection. |
| **Causal LLM Routing** (IBM) | **NeurIPS 2025** | IBM Research | End-to-end regret minimization from observational data. | Addresses partial observability but cross-model only. |
| **C3PO** (Valkanas et al.) | **NeurIPS 2025** | (Multi) | Self-supervised cascade with conformal prediction for probabilistic cost constraints. | Cost guarantees, but cross-model cascade. |
| **EAGLE-3** + **SuffixDecoding** | **NeurIPS 2025** | PKU+Microsoft / CMU+Snowflake | SOTA speculative decoding: 3-6.5x speedup (EAGLE-3), 5.3x on agentic (SuffixDecoding). | Provider-side optimization, not routing. |

**Takeaway**: All top-venue routing papers are cross-model. **No one does same-model cross-provider routing with cost+latency joint optimization.**

---

## B. Multi-Cloud & Spot Instance Serving (Most Directly Related Systems Work)

| Paper | Venue | Lab | Key Idea | Relevance |
|-------|-------|-----|----------|-----------|
| **SkyServe** (Mao, Stoica et al.) | **EuroSys 2025** | Berkeley Sky Lab | Serve AI models across regions/clouds with spot instances. SpotHedge policy. 43% cost savings, 2.1-2.3x better P50/P90/P99. | **Most directly related system.** Cross-cloud serving with hedging = our multi-provider problem. |
| **SkyPilot** (Yang, Stoica et al.) | **NSDI 2023** | Berkeley Sky Lab | Intercloud broker: auto-select best cloud/region/instance for ML workloads. | Foundation for cross-provider resource brokering. |
| **Can't Be Late** (Wu, Stoica et al.) | **NSDI 2024** (Outstanding Paper) | Berkeley Sky Lab | Spot scheduling with deadline guarantees. Uniform Progress policy. 27-84% cost savings. | Spot + on-demand fallback = cheap provider + expensive fallback. |
| **SpotServe** (Miao, Jia et al.) | **ASPLOS 2024** | CMU | First distributed LLM serving on spot instances. Dynamic parallelism, stateful recovery. 2.4-9.1x P99 reduction, 54% cost savings. | Spot-based LLM serving with migration. |
| **ServerlessLLM** (Fu et al.) | **OSDI 2024** | Edinburgh/NTU | Serverless LLM with fast checkpoint loading. 10-200x lower latency vs existing serverless. | Pay-per-use model enables different pricing. |
| **BlitzScale** (Zhang et al.) | **OSDI 2025** | (Multi) | O(1) host caching for live autoscaling. 94% lower tail latency vs ServerlessLLM. | Fast scaling affects provider reliability. |

**Takeaway**: Berkeley Sky Lab (SkyServe/SkyPilot/Can't Be Late) is the closest systems work to our problem. They solve cross-cloud serving but focus on training/general ML, not LLM inference routing with cost+latency SLOs.

---

## C. Disaggregated & Phase-Aware Serving (Shapes Provider Economics)

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **DistServe** (Zhong et al.) | **OSDI 2024** | Peking University | Disaggregate prefill/decoding onto separate GPU pools. 7.4x more requests or 12.6x tighter SLO. |
| **Splitwise** (Patel et al.) | **ISCA 2024** (Best Paper) | Microsoft/UW | Phase splitting onto different hardware. 1.4x throughput at 20% lower cost. |
| **Mooncake** (Qin et al.) | **FAST 2025** (Best Paper) | Moonshot AI (Kimi) | KVCache-centric disaggregation in production. 59-498% more request capacity. |
| **Sarathi-Serve** (Agrawal et al.) | **OSDI 2024** | Georgia Tech/MSRI | Chunked-prefill for precise throughput-latency control. |

**Why it matters**: Disaggregation means providers can price prefill and decode differently. A router should know which providers are prefill-optimized vs decode-optimized.

---

## D. Scheduling, Fairness & SLO (Determines Provider Latency Profiles)

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **Llumnix** (Sun et al.) | **OSDI 2024** | Alibaba | Runtime rescheduling via KV-cache migration. 10x better tail latency, 36% cost savings. |
| **Fairness in LLM Serving** (Sheng, Stoica et al.) | **OSDI 2024** | Berkeley LMSYS | Virtual Token Counter for fair sharing. 2x tight bound on service difference. |
| **SOLA** (Tsinghua) | **MLSys 2025** | Tsinghua | State-aware scheduling. SLO attainment 45.5% -> 99.4%. |
| **Aegaeon** (Xiang et al.) | **SOSP 2025** | PKU/Alibaba | Per-token GPU pooling for multi-model serving on the market. |

---

## E. Agentic Workloads (Validates Juncheng's Insight)

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **Sarathi-Serve** (Agrawal et al.) | Microsoft Research | Semantic Variable abstraction for multi-request LLM workflows. Up to 11.7x speedup. |
| **SGLang** (Zheng, Stoica et al.) | **NeurIPS 2024** | Berkeley LMSYS | Structured LLM programs with RadixAttention. Up to 6.4x throughput. |
| **Pie** (Gim et al.) | **SOSP 2025** | Yale | Programmable serving via inferlets. 1.3-3.4x on agentic workflows. |
| **LLMCompiler** (Kim et al.) | **ICML 2024** | Berkeley (Gholami) | Parallel function calling planner for agents. |
| **Chatbot Arena** (Chiang, Stoica et al.) | **ICML 2024** | Berkeley LMSYS | Open platform for LLM evaluation by human preference. Foundation for routing data. |

**Key empirical results** (arXiv, not yet published at top venue but important):
- **AgentCgroup** (arXiv 2602): Tool execution = **56-74%** of e2e latency. LLM = only 26-44%. **Validates Juncheng's insight that LLM is not the bottleneck.**
- **What Limits Agentic Efficiency?** (Bian et al., UW-Madison, arXiv 2510): Larger models can have *lower* latency than smaller ones due to API variability.

---

## F. Speculative Decoding (Changes Provider Cost-Latency Frontier)

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **EAGLE** (Li et al.) | **ICML 2024** | PKU/Microsoft | Feature-level speculation. 2.7-3.5x speedup on LLaMA2-70B. |
| **EAGLE-3** (Li et al.) | **NeurIPS 2025** | PKU/Microsoft | Direct token prediction + multi-layer fusion. 3.0-6.5x speedup. |
| **SpecInfer** (Miao, Jia et al.) | **ASPLOS 2024** | CMU Catalyst | Tree-based multi-draft speculation. 4.4x fewer decoding steps. |
| **Online Speculative Decoding** (Liu, Stoica, Zhang) | **ICML 2024** | Berkeley | Continuously adapt draft model to user distribution. |
| **Lookahead Decoding** (Fu, Stoica, Zhang) | **ICML 2024** | Berkeley | Parallel decoding without auxiliary models. Up to 1.8x. |
| **SuffixDecoding** (Oliaro, Jia et al.) | **NeurIPS 2025** (Spotlight) | CMU/Snowflake | Model-free spec decoding via suffix trees. 5.3x on agentic tasks. |

---

## G. KV Cache & Long-Context (Affects Per-Query Cost Structure)

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **CacheBlend** (Yao et al.) | **EuroSys 2025** (Best Paper) | UChicago | KV cache fusion for RAG. 2.2-3.3x TTFT reduction. |
| **LoongServe** (Wu et al.) | **SOSP 2024** | PKU | Elastic sequence parallelism for long contexts. |
| **Marconi** (Pan et al.) | **MLSys 2025** (Outstanding Paper) | Amazon/UW-Madison | Prefix caching for hybrid LLMs. 34.4x higher hit rates. |
| **IC-Cache** (Yu et al.) | **SOSP 2025** | (Multi) | In-context caching. 70% requests have similar past counterparts. 1.4-5.9x throughput. |

---

## H. Kernel & Quantization (Determines Provider Cost Floor)

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **FlashInfer** (Ye, Chen et al.) | **MLSys 2025** (Best Paper) | UW/CMU | Customizable attention engine. 29-69% ITL reduction. Backend for vLLM/SGLang. |
| **Hydragen** (Juravsky, Re, Mirhoseini) | **ICML 2024** | Stanford | Shared-prefix attention. Up to 32x throughput for batched shared-prefix. |
| **QServe** (MIT Han Lab) | **MLSys 2025** | MIT | W4A8KV4 quantization. A100-level throughput on 3x cheaper L40S. |
| **LServe** (MIT Han Lab/NVIDIA) | **MLSys 2025** | MIT/NVIDIA | Unified sparse attention for long sequences. 2.9x prefill speedup. |
| **SpInfer** (Wang et al.) | **EuroSys 2025** (Best Paper) | HKUST | Practical unstructured sparsity on GPUs. 1.58x e2e speedup at 30% sparsity. |
| **Atom** (Zhao, Chen et al.) | **MLSys 2024** | UW/CMU | Low-bit quantization. 7.7x throughput vs FP16. |
| **TidalDecode** (Yang, Jia et al.) | **ICLR 2025** | CMU | Position-persistent sparse attention. 2.1x decoding speedup. |
| **XGrammar** (Dong, Chen et al.) | **MLSys 2025** | CMU (Tianqi Chen) | Structured generation engine. 100x faster. Default in vLLM/SGLang. |

---

## I. Heterogeneous & Elastic Serving

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **Helix** (Mei et al.) | **ASPLOS 2025** | CMU Catalyst | Max-flow optimization for heterogeneous GPUs. 3.3x throughput, 66% less latency. |
| **ThunderServe** (Jiang et al.) | **MLSys 2025** | ETH Zurich/PKU | Scheduling for heterogeneous cloud. 2.1x throughput, 2.5x latency reduction. |
| **PowerInfer** (Song et al.) | **SOSP 2024** | SJTU | GPU-CPU hybrid on consumer hardware. 11.69x over llama.cpp. |
| **SuperServe** (Khare, Stoica, Tumanov) | **NSDI 2025** | Georgia Tech/Berkeley | Serve full latency-accuracy tradeoff via weight-shared super-networks. |
| **NanoFlow** (Zhu et al.) | **OSDI 2025** | UW | Intra-device parallelism via nano-batches. 1.91x throughput, 50-72% of theoretical peak. |

---

## J. Simulation & Benchmarking

| Paper | Venue | Lab | Key Idea |
|-------|-------|-----|----------|
| **Vidur** (Agrawal et al.) | **MLSys 2024** | MSRI/Georgia Tech | LLM inference simulator. <9% error. Finds optimal config in 1hr CPU vs 42K GPU-hrs. |
| **Scaling Inference-Efficient LMs** (Bian, Venkataraman) | **ICML 2025** | UW-Madison | Same-size models can differ 3.5x in latency. Modified Chinchilla laws for inference. |

---

## K. LLM Market Economics

| Paper | Venue | Authors | Key Idea |
|-------|-------|---------|----------|
| **The Emerging Market for Intelligence** | **NBER 2025** | Demirer, Fradkin, Tadelis, Peng | First empirical study of LLM market (OpenRouter+Azure). Open-source 90% cheaper. Price elasticity ~1. |
| **Economics of LLMs** | **ACM EC 2025** | Bergemann (Yale), Bonatti (MIT), Smolin | Optimal LLM pricing = menus of two-part tariffs. Formal mechanism design. |
| **Cloud Pricing: Spot Market Strikes Back** | **Management Science 2022** | Dierks, Seuken | Game theory + queueing for spot + on-demand. Market cannibalization analysis. |
| **Cloud Cost Optimization** | **Operations Research 2024** | Qu, Dawande, Janakiraman | Asymptotically optimal reserved vs on-demand strategy. O(1/sqrt(theta)) gap. |
| **Capacity Reservation for Demand Surges** | **POM 2024** | Chen, Lei, Moinzadeh | Newsvendor-type solution for bursty cloud demand. Secondary market value. |
| **Cloud Server Deployment** | **M&SOM 2025** | Liu et al. (MIT/Microsoft) | Two-stage stochastic programming for server deployment under uncertainty. |
| **State of AI (100T Tokens)** | **arXiv 2601** (a16z+OpenRouter) | Aubakirova, Atallah et al. | Agentic inference = fastest growing. Providers: 27->90. Programming: 11%->50%+ of volume. |
| **Inference Economics** | **arXiv 2506** (Epoch AI) | Erdil | Cost-speed Pareto frontiers. Explains provider price/latency tradeoff diversity. |

---

## Summary: What's Missing (Our Opportunity)

### Gap 1: Same-Model Multi-Provider Routing
- **RouteLLM/FrugalGPT/AutoMix** (ICLR/ICML/NeurIPS): All route across *different models*
- **SkyServe/SpotServe** (EuroSys/ASPLOS): Route across *clouds/regions* but for general ML, not LLM inference SLOs
- **Nobody**: Routes same model across providers with joint cost+latency optimization

### Gap 2: Capacity Block Pricing for LLM APIs
- **OR literature** (Qu et al., Chen et al.): Mature theory for cloud reserved vs on-demand
- **Nobody**: Applies stochastic programming to LLM API capacity procurement
- **Two-Level Ski-Rental** (AAAI 2024): Closest theoretical framework (rent/buy/combo)

### Gap 3: Agentic-Aware Routing
- **AgentCgroup**: Tool calls = 56-74% of latency
- **Parrot/SGLang/Pie** (OSDI/NeurIPS/SOSP): Optimize serving but not *cross-provider* routing
- **Nobody**: Routing that accounts for tool-call-dominated latency structure

### Papers to Cite in Related Work (Prioritized)

**Must Cite:**
- RouteLLM (ICLR'25), FrugalGPT (TMLR'24), Unified Routing (ICML'25), AutoMix (NeurIPS'24)
- SkyServe (EuroSys'25), SpotServe (ASPLOS'24), SkyPilot (NSDI'23)
- DistServe (OSDI'24), Splitwise (ISCA'24)
- Emerging Market for Intelligence (NBER'25), Cloud Pricing (MS'22)

**Should Cite:**
- Compound AI Scaling (NeurIPS'24), Archon (ICML'25), Causal Routing (NeurIPS'25)
- SGLang (NeurIPS'24), Parrot (OSDI'24), Llumnix (OSDI'24)
- Can't Be Late (NSDI'24), ServerlessLLM (OSDI'24)
- Cloud Cost Optimization (OR'24), Economics of LLMs (ACM EC'25)
- Vidur (MLSys'24), FlashInfer (MLSys'25)

**Good Context:**
- EAGLE-3 (NeurIPS'25), SuffixDecoding (NeurIPS'25)
- Scaling Inference-Efficient LMs (ICML'25)
- QServe (MLSys'25), Helix (ASPLOS'25), ThunderServe (MLSys'25)
