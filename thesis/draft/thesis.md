# Introduction

TODO

# Preliminaries

## Notation

## Linear Programming

*[What is optimization and optimization problem?]*

**Definition** (Optimization problem) A (mathematical) optimization problem is
is a problem of the form
$$
\begin{align*}
    \text{minimize}\;&  f(x) \\
    \text{subject to}\;&    g_i(x) \le b_i \quad i = 1, 2, \ldots, m
\end{align*}
$$
where $n \in \mathbb{N}$, $x \in \mathbb{R}^n$ is a vecfor of *optimization variables* , the function $f : \mathbb{R}^n \to \mathbb{R}$ is called the *objective function* and the functions $g_i$ are *constraint* functions for some *bounds* $b_i$. 

Note that equality constraints can be expressed as two inequalities.

**Definition** (Feasible solution) Given a problem of mathematical optimization (as defined above), the vector $x \in \mathbb{R}^n$ is said to be *feasible* if it satisfies *all* constraints $g_i$.

**Definition** (Optimal solution) Given a problem of mathematical optimization (as defined above), the vector $x \in \mathbb{R}^n$ is said to be *optimal* if has the yields the smallest objective value among all feasible vectors, i.e.
$$
f(x) \ge f(x^*)
$$
for $x \in \mathbb{R}$ feasible.

[boyd]

*[What is linear programming?]*

Optimization problems are additionally classified into more specific classes based on the form of the objective and constraint functions. One such class is linear programming.

*[ How is a LP defined? ]*

**Definition** (Linear Programming Problem) A *Linear Programming (LP) problem*
is a problem of mathematical optimization, such that the objective and constraint functions are linear, i.e, satisfy
$$
    f(\alpha x + \beta y ) = \alpha f(x) + \beta f(y)
$$

Function application with linear functions can be generally represented as a matrix vector multiplication. Therefore, we get that LP problems can be expressed in the following way:

[TODO] Integer Linear Programming

$$
\begin{align*}
    \text{minimize}\;&  c^T x \\
    \text{subject to}\;&    A x \le b
\end{align*}
$$
where $c \in \mathbb{R}^n$ is (TODO ???), $A \in \mathbb{R}^{m \times n}$ and
$b = (b_1, \ldots, b_m) \in \mathbb{R}^m$.

[TODO] Standard and Canonical form
[TODO] Variable bounds form

*[ When was the concept of LP conceived and by whom? ]*

[TODO] Unimportant?

*[ Why is LP interesting/usefull/important? ]*

Linear programming is one of the most studied classes of optimization problems.
The reason is that a great number of real world problems can be formulated
as a linear programming problem. Linear programming has been heavily used in in
microeconomics and company management, such as planning, production,
transportation, decision making and other issues, either to maximize the income
or minimize the costs of a production scheme.

Moreover, many real world problems if not necessarily linear in nature, can be
approximated by linear models.

Many combinatorial problems can be expressed as linear programs, for examaple
shortest paths, maximum flow, maximum matching, TSP (integer LP) and many others.

Linear programming is also used as a subroutine in more complex algorithms.
Examples including approximation algorithms (LP SAT, Scheduling), optimization
algorithms (Frank-Wolfe) 

[TODO] Theoretical results (duality)

However, as LP is a generic tool, there often exist specialized algorithms for a
given problem offering better performance. It is then usefull to use LP if such
algorithm is not known or devising such algorithm is too costly.

*[ Where to find detailed descriptions of linear programming ]*

For a more detailed discussion of LP see:
[ Matousek, Chvatal ]

## LP algorithms

[./simplex_algorithm_overview]


### Standard Simplex Algorithm

[TODO] Computational form

Review dictionary "student" algorithm

- How does Standard Simplex work?

### Revised Simplex Algorithm

*[How does Revised Simplex work?]*

**Step 0** (Initialization).

Presolve the LP problem.
Scale the LP problem.
Select an initial basic solution (B, N).
if the initial basic solution is feasible then proceed to step 2.

**Step 1** (Phase I).

Construct an auxiliary problem 

Apply the revised simplex algorithm. If the final basic solution (B, N) is feasible, then proceed to step 2 in order to solve the initial problem. The LP problem can be either optimal or infeasible.

**Step 2** (Phase II).

**Step 2.1** (Test of Optimality).

Choose the index l of the entering variable using a pivoting rule. Variable x_l enters the basis.

**Step 2.2**(Pivoting).

Compute the pivot column $h_l$

Choose the leaving variable

**Step 2.3** (Update).

Swap indices k and l. Update the new basis inverse AB , using a basis update
scheme. Update vectors $xB$ , $w$, and $s_N$ .


*[TODO: Revised simplex vs standard simplex]*

### Effective Implementation of Revised Simplex

- There exist effective implementations of Simplex (however mostly floating point)
- exact LP solutions


#### Sparse Matrices

- solving sparse system of linear equations
- FTRAN, BTRAN

#### LU decomposition

- Markowitz rule
- fill-in

#### Matrix decompositions

- Bertels-Golub
- Forrest-Tomlin
- Suhl-Suhl

####  Initial basis computation

#### Presolving


# Experimental design

TODO

# Implementation

### Tentative solution in Python

- current options for exact LP solvers
- python, numpy, sympy, sage, GLPK, Gurobi, Floating point exact


### GLPK exact trace API

- my work 

## MCF

- Mares, Hladik
- Total unimodularity
- Effective combinatorial algorithms

## MMCF

The multicommodity min-cost flow (MMCF) problem is a generalization of the
ordinary single-commodity min-cost flow (MCF) problem, in which flows of a
different nature (commodities) must be routed at min- imal cost on a network,
competing for the resources represented by the arc capacities.

Although MMCF is a structured linear program (LP), standard LP techniques often
fail to be efficient enough in practice, and several specialized algorithmshave
been proposed for its solution during the last four decades.

Let us remark that the number of commodities in real-life MMCFs ranges from
just a few, as in most dis- tribution problems, to very many, such as the
number of all the possible O/D pairs in some telecommunica- tion models. This
is of course crucial in the choice of the most suitable solution algorithm
(Frangioni and Gallo 1999)

- MCF vs MMCF
    * No known combinatorial algorithm for MMCF

# Results

## MMCF Problem sets
- http://groups.di.unipi.it/optimize/Data/MMCF.html

## NETLIB

Description of netlib problems

# Conclusions

TODO

# Appendix

## Running the tests

- docker
- Graph
