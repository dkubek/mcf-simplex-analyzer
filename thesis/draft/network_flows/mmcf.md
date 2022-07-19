# MMCF

[aola Cappanera, Antonio Frangioni]

The multicommodity min-cost flow (MMCF) problem is a generalization of the ordinary single-commodity min-cost flow (MCF) problem, in which flows of a different nature (commodities) must be routed at minimal cost on a network, competing for the resources represented by the arc capacities.

Although MMCF is a structured linear program (LP), standard LP techniques often fail to be efficient enough in practice, and several specialized algorithms have been proposed for its solution. [TODO][What algorithms?]

The number of commodities in real-life MMCFs ranges from just a few, as in most distribution problems, to very many, such as the number of all the possible O/D pairs in some telecommunication models. 
(Frangioni and Gallo 1999)


## MMCF as a LP

$$
\begin{align*}
    \text{minimize}\;&  \sum_{h = 1}^{k} \sum_{e} \nu_{e}^h f_e^h \\
    \text{subject to}\;& \sum_{uv} - f_{vu}^h = d_v^k && \forall\, v, h\\
    & 0 \le f_{uv}^h \le c_{uv}^h && \forall\; u,v,h && \text{(individual capacity constraint)}\\
    & \sum_h f_{uv}^h \le c_{uv} && \forall\; u,v && \text{(mutual capacity constraint)}
\end{align*}
$$
[TODO][Can be described below maybe...]


- MCF vs MMCF
    * No known combinatorial algorithm for MMCF
    * "easiest" such problem