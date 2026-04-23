\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}

% Stage 1 section (include in main with \input{docs/algorithm/offline_stage1_new})
\section{Offline Routing with Fixed-Window Subscriptions (Stage 1)}
\label{sec:offline-stage1}

This section formalizes the Stage~1 offline optimization problem.
We consider a hybrid inference system leveraging heterogeneous commercial subscriptions (e.g., ChatGPT Plus, Claude Pro) alongside pay-per-token APIs.
The objective is to minimize total marginal API cost by allocating scarce subscription quota to the highest-value requests.

\subsection{Problem Overview}
Commercial subscriptions typically enforce rate limits over rolling windows (e.g., \(160\) requests per \(3\) hours).
To derive a tractable closed-form oracle, we approximate these rolling limits using \emph{disjoint, fixed time windows} aligned to a reference time.
Crucially, each subscription acts as a \emph{shared scarce resource} where distinct models compete for the same quota.
Since fixed windows are strictly less restrictive than rolling windows (quota resets instantly at boundaries), this oracle provides an \emph{optimistic lower bound} on the achievable API cost under the true rolling-window constraints.

\subsection{System Model}
\paragraph{Requests.}
We are given a request trace \(\mathcal{R} = \{1,\dots,n\}\). Each request \(i\) is characterized by:
\begin{itemize}
  \item Arrival time \(a_i\) (seconds since epoch);
  \item Model type \(m(i)\) from a model set \(\mathcal{M}\);
  \item Input tokens \(n_i^{\text{in}}\) and output tokens \(n_i^{\text{out}}\).
  \item Potential API cost \(c_i^{\text{API}}\) if served by the pay-per-token provider;
  \item Quota consumption weight \(w_i\) if served by a subscription (typically \(w_i=1\), but generalized to support alternative quota definitions).
\end{itemize}

\paragraph{API costs.}
The pay-per-token API \(S_A\) has fixed per-model prices \(p_m^{\text{in}}, p_m^{\text{out}}\) per 1K tokens. The API cost for request \(i\) is
\[
  c_i^{\text{API}} = \frac{n_i^{\text{in}}}{1000}\,p_{m(i)}^{\text{in}} + \frac{n_i^{\text{out}}}{1000}\,p_{m(i)}^{\text{out}}.
\]

\paragraph{Subscription plans.}
Let \(\mathcal{P}\) be the set of subscription plans (e.g., an OpenAI plan and an Anthropic plan).
Each plan \(p \in \mathcal{P}\) is characterized by a window length \(W_p\), a per-window request quota \(Q_{\text{win},p}\), and a window alignment time \(t_{0,p}\).
Define the window index function:
\[
  \operatorname{win}_p(t) = \left\lfloor \frac{t - t_{0,p}}{W_p} \right\rfloor.
\]
In each window \(k\), plan \(p\) can serve at most \(Q_{\text{win},p}\) requests at zero marginal cost; quota does not carry over.
Daily and weekly quotas are special cases with \(W_p \in \{24\text{ hours}, 7\text{ days}\}\) and \(t_{0,p}\) aligned to the corresponding calendar boundary (in a chosen timezone).

\paragraph{Compatibility groups and model hierarchy.}
Real-world subscriptions only cover subsets of models. We assume each request is compatible with at most one plan.
Let \(p(i) \in \mathcal{P} \cup \{\varnothing\}\) denote the plan compatible with request \(i\); if \(p(i)=\varnothing\), the request must use the API.
In our experiments, we treat the following plan-level model groups:
\begin{itemize}
  \item \textbf{OpenAI plan:} \(\{ \texttt{gpt-5}, \texttt{gpt-5.1} \}\).
  \item \textbf{Anthropic plan:} \(\{ \texttt{claude-4.5-opus}, \texttt{claude-4.5-sonnet}, \texttt{claude-4.1-opus}, \texttt{claude-4.1-sonnet} \}\).
\end{itemize}
Within each plan, models form an implicit hierarchy driven by their avoided API cost: routing a high-cost request (e.g., \texttt{claude-4.5-opus}) to the subscription typically yields higher savings than routing a low-cost request (e.g., \texttt{claude-4.1-sonnet}).

\paragraph{Alternative quota interpretation.}
The quota weight \(w_i\) captures \emph{alternative} subscription accounting schemes. For standard message-based plans, \(w_i=1\).
More generally, \(w_i\) can model token-based quotas or implicit throttling where premium models consume more quota units.
In our current evaluation, we use the standard case \(w_i=1\) and introduce \(w_i\) to keep the formulation future-proof.

\paragraph{Assumptions.}
\begin{itemize}
  \item Subscription fees are sunk costs and omitted from the objective.
  \item \(S_Q\) has no concurrency bottleneck; only per-window request quotas matter.
  \item Requests are independent; there are no ordering constraints.
\end{itemize}

\subsection{Optimization Problem}
For each request \(i\), define binary routing variables \(x_i^Q, x_i^A \in \{0,1\}\) with the assignment constraint \(x_i^Q + x_i^A = 1\).
If \(p(i)=\varnothing\), we require \(x_i^Q = 0\).

The objective is to minimize total API spending:
\[
  \min\; \sum_{i\in\mathcal{R}} x_i^A\,c_i^{\text{API}}.
\]
Equivalently, since \(x_i^A = 1 - x_i^Q\), this is the same as maximizing total savings from quota usage:
\[
  \max\; \sum_{i\in\mathcal{R}} x_i^Q\,c_i^{\text{API}}.
\]

For each plan \(p\) and window \(k\), define the set of requests compatible with \(p\) that arrive in window \(k\):
\[
  \mathcal{R}_{k,p} = \{ i \in \mathcal{R} \mid p(i)=p \land \operatorname{win}_p(a_i)=k \}.
\]
Subject to per-window quota constraints:
\[
  \sum_{i\in\mathcal{R}_{k,p}} w_i\,x_i^Q \le Q_{\text{win},p}, \quad \forall p \in \mathcal{P},\; \forall k.
\]

\subsection{Closed-Form Optimal Solution}
Because windows are disjoint and plans are independent under the single-plan compatibility assumption, the problem decomposes by each \((p,k)\).
Within each plan-window \((p,k)\), we solve a constrained selection problem:
\begin{itemize}
  \item \textbf{Standard case (\(w_i=1\)):} the problem reduces to a cardinality-constrained selection problem, and sorting by \(c_i^{\text{API}}\) is optimal.
  \item \textbf{Generalized case (variable \(w_i\)):} the problem becomes a \(0/1\) knapsack instance and is NP-hard in general.
\end{itemize}

\paragraph{Algorithm (standard case \(w_i=1\)).}
For each plan \(p\) and window \(k\):
\begin{enumerate}
  \item Collect requests \(\mathcal{R}_{k,p}\).
  \item Sort \(\mathcal{R}_{k,p}\) in descending order by \(c_i^{\text{API}}\).
  \item Route the top \(\min(Q_{\text{win},p}, |\mathcal{R}_{k,p}|)\) requests to \(S_Q\), and route the remaining requests to \(S_A\).
\end{enumerate}

\paragraph{Proof sketch (standard case \(w_i=1\)).}
Within each \((p,k)\), selecting the top-\(Q_{\text{win},p}\) requests by \(c_i^{\text{API}}\) is optimal because all weights are equal; any solution that assigns quota to a lower-cost request while sending a higher-cost request to the API can be improved by swapping them.

\paragraph{Generalized case (variable \(w_i\)).}
When \(w_i\) varies, each \((p,k)\) subproblem is a \(0/1\) knapsack instance and is NP-hard.
An offline oracle can be instantiated using a MIP solver, or pseudo-polynomial dynamic programming when capacities are small integers.
For intuition, the fractional relaxation suggests prioritizing high value-to-weight ratio \(c_i^{\text{API}}/w_i\); however, this ratio-based greedy rule is not guaranteed optimal for the \(0/1\) problem.

\paragraph{Opportunity cost interpretation.}
Subscription quota is a scarce resource. Routing a low-value request to \(S_Q\) incurs an \emph{opportunity cost} by displacing a higher-value request that could have been served for free.
In the standard case \(w_i=1\), the opportunity cost of consuming one additional quota slot is the avoided API cost of the best unselected request (i.e., the \((Q_{\text{win},p}+1)\)-th order statistic within \(\mathcal{R}_{k,p}\)).
In the generalized case, the LP relaxation yields a shadow price per quota unit (a Lagrange multiplier) that plays the same role.

\subsection{Multi-Model Example (Shared Quota)}
Consider the \textbf{Anthropic plan} shared by a hierarchy of models. Suppose in a single window we receive:
\begin{itemize}
  \item \(20\) requests for \texttt{claude-4.5-opus} (cost: \$1.50);
  \item \(50\) requests for \texttt{claude-4.5-sonnet} (cost: \$0.30);
  \item \(30\) requests for \texttt{claude-4.1-sonnet} (cost: \$0.05).
\end{itemize}
If \(Q_{\text{win}}=45\) and \(w_i=1\), the algorithm sorts all \(100\) requests by \(c_i^{\text{API}}\), assigns all \(20\) \texttt{claude-4.5-opus} requests to \(S_Q\), then fills the remaining \(25\) slots with the most expensive \texttt{claude-4.5-sonnet} requests. The rest are routed to \(S_A\).

\paragraph{Insight.}
This illustrates the opportunity cost of subscription slots: routing a low-value request (e.g., \texttt{claude-4.1-sonnet}) to the subscription is suboptimal if it displaces a high-value request (e.g., \texttt{claude-4.5-opus}).
Under this view, models effectively ``bid'' for scarce quota using their avoided API cost.

\subsection{Notes and Caveats}
\begin{itemize}
  \item \textbf{Fixed-window alignment matters:} the lower bound depends on the choice of \(t_{0,p}\). Boundary bursts may be feasible under disjoint windows but infeasible under rolling windows.
  \item \textbf{Alternative quota definitions:} if subscription limits are measured in tokens (or credits) rather than request counts, Stage~1 becomes a general knapsack problem and the sorting-based rule is no longer guaranteed optimal.
  \item \textbf{Multiple eligible plans:} if a request can be served by multiple plans, the global problem no longer decomposes by plan and requires a joint allocation across plans.
\end{itemize}

\end{document}
