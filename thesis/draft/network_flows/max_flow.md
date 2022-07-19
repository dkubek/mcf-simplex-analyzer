# Max Flow

[Mares (Pruvodce, Prochazkou), Hladik]

**Definition (Network)**
Network is the 5-tuple $(V, E, s, t, c)$ where

- $(V, E)$ is a oriented graph
- $c : E \to \mathbb{R}^+$ is a function assigning *capacities* to edges
- $s, t$ are vertices of the graph. The vertex $s$ is called *source* or *origin* and $t$ is called *target* or *destination*.

**Definition (Inflow, Outflow, Excess)**
Given network $(V, E, s, t, c)$ and function $f : E \to \mathbb{R}$. For $v \in V$ we define

*Inflow* 
$$
f^+(v) = \sum_{e = (\cdot, v)} f(e)
$$

*Outflow* 
$$
f^-(v) = \sum_{e = (v, \cdot)} f(e)
$$

*Excess* 
$$
f^\Delta(v) = f^+(v) - f^-(v)
$$


**Definition (Flow)**
*Flow* in the given network is a function $f : E \to \mathbb{R}$, such that:
1. The flow respects capacities 
    $$(\forall e \in E)\; 0 \le f(e) \le c(e)$$
2. Kirchhoffs law
$$
(\forall v \in V - \{s, t\})\; f^{\Delta}(v) = 0
$$

What we are interested is *maximum network flow*, i.e. the maximum quantity we can push through the network.

We can formulate the problem also as a linear program

$$
\begin{align*}
    \text{maximize}\;&  f^{\Delta}(t) \\
    \text{subject to}\;&    f^{\Delta}(e) = 0 \\
    & 0 \le f_e \le c_e \quad \forall e \in E
\end{align*}
$$


[TODO][Total unimodularity]
Total unimodularity (network problems have the property that it is possible to find a integral solution in polynomial time)

[TODO][Combinatiorail algorithms]
Effective combinatorial algorithms (FF (with shortest path), Dinic, Preflow Push)
