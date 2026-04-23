\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}
\section{Offline Routing with Dual Subscriptions (Stage 2)}
\label{sec:offline-stage2}

This section formalizes Stage~2, which augments Stage~1 by adding a
concurrency-limited subscription \(S_C\) alongside the daily-quota subscription
\(S_Q\) and the pay-per-token API \(S_A\). The objective remains to minimize the
marginal API cost given full knowledge of the request trace, but routing and
scheduling decisions now interact through concurrency.

\subsection{Problem Overview and Relation to Stage~1}
Stage~1 provides a base case with only \(S_Q\) and \(S_A\); it admits a sorting-based
optimal solution because each request consumes one unit of quota. Stage~2 adds
\(S_C\), where requests consume time on a limited number of concurrent slots.
This coupling between count-limited and time-limited resources makes the problem
substantially more complex.

\subsection{System Model}
We consider a set of requests \(\mathcal{R} = \{1,\dots,n\}\). Each request \(i\) has:
\begin{itemize}
  \item Arrival time \(a_i\) (seconds).
  \item Model type \(m(i)\) from model set \(\mathcal{M}\).
  \item Tokens \((n_i^{\text{in}}, n_i^{\text{out}})\).
  \item Estimated duration \(p_i\) (seconds) if executed on \(S_C\).
  \item Optional deadline \(d_i\) (hard SLO), e.g., \(d_i = a_i + L_i\).
\end{itemize}

Providers:
\begin{itemize}
  \item Daily-quota subscription \(S_Q\): at most \(Q\) requests per calendar day; zero marginal cost.
  \item Concurrency-limited subscription \(S_C\): at most \(C\) requests executing simultaneously; zero marginal cost. Requests assigned here occupy one slot for \(p_i\) seconds.
  \item API \(S_A\): unlimited capacity with fixed per-token prices per model; request \(i\) costs
        \[
          c_i^{\text{API}} = \frac{n_i^{\text{in}}}{1000}\,p_{m(i)}^{\text{in}} + \frac{n_i^{\text{out}}}{1000}\,p_{m(i)}^{\text{out}}.
        \]
\end{itemize}

Assumptions:
\begin{itemize}
  \item Subscription fees are sunk and excluded from the objective.
  \item \(S_Q\) is modeled purely as a daily counter (concurrency is not the bottleneck).
  \item \(S_C\) is modeled purely as instantaneous concurrency (no daily cap).
  \item Requests are independent; no preemption on \(S_C\) (once started, a job runs for \(p_i\)).
  \item Deadlines are hard constraints (a schedule violating \(d_i\) is infeasible).
\end{itemize}

\subsection{Optimization Problem}
For each request \(i\), define binary routing variables \(x_i^Q, x_i^C, x_i^A \in \{0,1\}\) with
assignment constraint \(x_i^Q + x_i^C + x_i^A = 1\). If \(x_i^C=1\), let \(s_i\) be the start
time on \(S_C\).

Objective:
\[
  \min \sum_{i\in\mathcal{R}} x_i^A\,c_i^{\text{API}}.
\]
Subject to:
\begin{align}
  &\text{(Quota)} && \sum_{i: \operatorname{day}(a_i)=d} x_i^Q \le Q, \quad \forall d, \\
  &\text{(Concurrency)} && \sum_{i\in\mathcal{R}} \mathbf{1}[s_i \le t < s_i + p_i] \cdot x_i^C \le C, \quad \forall t, \\
  &\text{(Arrival)} && s_i \ge a_i \quad \text{whenever } x_i^C=1, \\
  &\text{(Deadline)} && s_i + p_i \le d_i \quad \text{whenever } x_i^C=1 \text{ and } d_i \text{ exists}.
\end{align}

\subsection{Complexity}
Stage~2 couples a count-limited allocation (\(S_Q\)) with a time-indexed machine
scheduling (\(S_C\)). It generalizes knapsack and parallel-machine scheduling with
release times and deadlines; the resulting problem is NP-hard.

\subsection{Practical Offline Oracle Design}
We present two offline modeling choices to instantiate the oracle used for evaluation.

\subsubsection{Time Discretization (Slot-Based ILP)}
\label{sec:slot-ilp}

We adopt a time-indexed formulation by discretizing the time horizon into slots $t \in \{0, 1, \dots, T\}$ of width $\Delta$ (e.g., $\Delta=1s$). This transforms the complex scheduling problem into a standard packing problem solvable by off-the-shelf MIP solvers. The horizon $T$ is set to cover the latest deadline among all requests (or the trace end time if no deadline exists).

\textbf{Discretized Parameters.} We map continuous parameters to discrete slots. Let $\hat{p}_i = \lceil p_i/\Delta \rceil$ be the processing duration in slots. The valid start window for request $i$ on $S_C$ is defined by:
\begin{itemize}
    \item Earliest start slot: $\alpha_i = \lceil a_i/\Delta \rceil$
    \item Latest valid start slot: $\beta_i = \lfloor d_i/\Delta \rfloor - \hat{p}_i$
\end{itemize}
If a request $i$ has no hard deadline, we set $d_i$ to a sufficiently large value (e.g., end of day). If $\alpha_i > \beta_i$, the request cannot be scheduled on $S_C$.

\textbf{Variables.} We retain the routing variables $x_i^Q, x_i^C, x_i^A \in \{0,1\}$ from the original problem. Additionally, we introduce binary scheduling variables $z_{i,t}$ for $S_C$, defined only within the valid window $[\alpha_i, \beta_i]$:
\begin{equation}
    z_{i,t} \in \{0, 1\}, \quad \forall i \in \mathcal{R}, \ t \in [\alpha_i, \beta_i].
\end{equation}
Here, $z_{i,t}=1$ implies request $i$ starts processing on $S_C$ exactly at time slot $t$.

\textbf{Optimization Problem.} The ILP formulation is:

\begin{align}
    \text{Minimize} \quad & \sum_{i \in \mathcal{R}} x_i^A c_i^{API} \label{eq:obj} \\
    \text{Subject to} \quad & x_i^Q + x_i^C + x_i^A = 1, \quad \forall i \in \mathcal{R} \label{eq:assign} \\
    & \sum_{i: \text{day}(a_i)=d} x_i^Q \le Q, \quad \forall d \label{eq:quota} \\
    & \sum_{t=\alpha_i}^{\beta_i} z_{i,t} = x_i^C, \quad \forall i \in \mathcal{R} \label{eq:consistency} \\
    & \sum_{i \in \mathcal{R}} \sum_{\tau=\max(\alpha_i, t-\hat{p}_i+1)}^{\min(\beta_i, t)} z_{i,\tau} \le C, \quad \forall t \in \{0, \dots, T\} \label{eq:concurrency}
\end{align}

Constraint (\ref{eq:assign}) ensures each request is routed to exactly one provider. Constraint (\ref{eq:consistency}) links the routing decision $x_i^C$ to the scheduling variables $z_{i,t}$. Constraint (\ref{eq:concurrency}) enforces the concurrency limit: at any slot $t$, the number of active requests on $S_C$ cannot exceed $C$.
\subsubsection{Alternative Formulation: Event-Driven Models}

An alternative approach to discretization is to use \textit{event-driven continuous-time formulations}. These models represent time via specific event points (e.g., request arrivals and completions) rather than a uniform grid. Theoretically, this approach eliminates discretization errors and can yield more compact models when the workload is sparse over a long horizon.

However, enforcing concurrency constraints in continuous time typically requires modeling job ordering on finite resources, which introduces significant formulation complexity regarding sequencing variables.

\end{document}
