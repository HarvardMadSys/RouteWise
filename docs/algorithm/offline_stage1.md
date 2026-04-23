\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}


% Stage 1 section (include in main with \input{docs/algorithm/offline_stage1})
\section{Offline Routing with Single Subscription (Stage 1)}
\label{sec:offline-stage1}

This section formalizes the Stage~1 offline optimization problem for our hybrid inference system.
Stage~1 considers a daily-quota subscription (\(S_Q\)) and a pay-per-token API (\(S_A\)).
The objective is to compute the minimum possible API cost for a given request trace when the daily quota is known in advance.

\subsection{Problem Overview}
We consider a single system serving one or more models. Each request can be served either by a daily-quota subscription \(S_Q\) (zero marginal cost, limited by a daily count) or by a pay-per-token API \(S_A\) (unlimited capacity, fixed per-model pricing). The subscription fee is treated as sunk and excluded from the objective.

\subsection{System Model}
\paragraph{Requests.} We are given a set of requests \(\mathcal{R} = \{1,\dots,n\}\). Each request \(i\) has:
\begin{itemize}
  \item Arrival time \(a_i\) (seconds since epoch);
  \item Model type \(m(i)\) from a model set \(\mathcal{M}\);
  \item Input tokens \(n_i^{\text{in}}\) and output tokens \(n_i^{\text{out}}\).
\end{itemize}
\paragraph{Providers.}
\begin{itemize}
  \item Daily-quota subscription \(S_Q\): at most \(Q\) requests per calendar day; zero marginal cost.
  \item Pay-per-token API \(S_A\): unlimited capacity; fixed per-model prices \(p_m^{\text{in}}, p_m^{\text{out}}\) per 1K tokens. The API cost for request \(i\) is
  \[
    c_i^{\text{API}} = \frac{n_i^{\text{in}}}{1000}\,p_{m(i)}^{\text{in}} + \frac{n_i^{\text{out}}}{1000}\,p_{m(i)}^{\text{out}}.
  \]
\end{itemize}
\paragraph{Assumptions.}
\begin{itemize}
  \item Subscription fees are sunk costs and omitted from the objective.
  \item \(S_Q\) has no concurrency bottleneck; only the daily count \(Q\) matters.
  \item API pricing is fixed per model; we do not model multiple competing APIs.
  \item Requests are independent; there are no ordering constraints.
\end{itemize}

\subsection{Optimization Problem}
For each request \(i\), define binary routing variables \(x_i^Q, x_i^A \in \{0,1\}\) with the assignment constraint \(x_i^Q + x_i^A = 1\). The objective is to minimize total API spending:
\[
  \min\; \sum_{i\in\mathcal{R}} x_i^A\,c_i^{\text{API}}.
\]
Subject to the daily quota constraint for every calendar day \(d\):
\[
  \sum_{i:\,\operatorname{day}(a_i)=d} x_i^Q \le Q.
\]

\subsection{Closed-Form Optimal Solution}
Since each request consumes one unit of quota on \(S_Q\), Stage~1 reduces to a knapsack problem with identical item weights. The optimal solution is obtained by sorting requests by API cost in descending order per day and assigning the top \(Q\) to \(S_Q\); remaining requests use \(S_A\).

\paragraph{Algorithm.}
For each day \(d\):
\begin{enumerate}
  \item Collect requests \(\mathcal{R}_d = \{ i\in\mathcal{R} : \operatorname{day}(a_i)=d \}\).
  \item Sort in descending order by \(c_i^{\text{API}}\).
  \item Assign the top \(\min(Q, |\mathcal{R}_d|)\) requests to \(S_Q\); assign the rest to \(S_A\).
\end{enumerate}
\paragraph{Proof of Optimality.}
Selecting the top-\(Q\) items by value is optimal when all weights are equal; any deviation can be improved by swapping a lower-cost request using \(S_Q\) with a higher-cost request using \(S_A\).
\paragraph{Complexity.}
Sorting dominates: \(\mathcal{O}(n\log n)\) time and \(\mathcal{O}(n)\) space.

\subsection{Multi-Model with a Shared Quota}
When a single \(S_Q\) is shared across multiple models, Stage~1 remains unchanged conceptually. For each day, pool all requests across models and sort by their per-request API cost \(c_i^{\text{API}}\). Assign the top \(Q\) to \(S_Q\).

\paragraph{Example.}
Suppose the system serves both DeepSeek (high API cost) and Qwen (low API cost), and both are eligible for the shared quota. DeepSeek requests will have larger \(c_i^{\text{API}}\) and appear earlier in the sorted list. Consequently, the quota is filled with DeepSeek first; Qwen requests use quota only if capacity remains.


\end{document}
