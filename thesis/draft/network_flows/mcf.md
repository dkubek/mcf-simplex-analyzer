# Minimum-cost flow

[https://feog.github.io/chap3dm.pdf]

Generalization of max flow. Can be solved efficiently using the *network simplex algorithm*.


**Definition (Minimum Cost Flow Network)**
Let
 - $G = (V, E)$ be a directed graph
 - $c : E \to \mathbb{R}$ be a *capacity* function
 - $\nu : E \to \mathbb{R}$ be a *cost* function
 - $d : V \to \mathbb{R}$ be a *demand* function and it holds that
 $$ \sum_{v \in V} d(v) = 0 $$
Then $G$ together with the functions $c$, $\nu$, $d$ is called
*minimum cost flow network*.

When
- $d(v) > 0$ we refer to $v$ as *supply* node
- $d(v) < 0$ we refer to $v$ as *demand* node
- $d(v) = 0$ we refer to $v$ as *transit* node



[TODO][Reduction MCF to MF]

MCF -> MF
Maximum flow problem. Let all nodes have zero demand, and let the cost associated with traversing any edge be zero. Now, introduce a new edge (t,s) from the current sink t to the current source s. Stipulate that the per-unit cost of sending flow across edge (t,s) equals -1, and permit (t,s) infinite capacity. [https://www.wikiwand.com/en/Minimum-cost_flow_problem]


## MCF as a LP

$$
\begin{align*}
    \text{minimize}\;&  \sum_{e} \nu(e)f(e) \\
    \text{subject to}\;&    f^{\Delta}(v) = 0 \\
    & f(u, v) = - f(v, u)  && \text{(skew-symmetry)}\\ 
    & \sum_{sw \in E} f(sw) = d && \text{(skew-symmetry)} \\
    & \sum_{wt \in E} f(wt) = d \\
    & 0 \le f_e \le c_e \quad \forall e \in E
\end{align*}
$$