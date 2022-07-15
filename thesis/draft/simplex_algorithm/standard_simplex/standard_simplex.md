[TODO][Standard Simplex - Dictionary]

In this section we will describe the Standard Simplex algorithm in the
dictionary form. We will also explain why it is not suitable for large or sparse
problems and where it falls short.

Suppose we are given problem in standard form.

## Initialization

[TODO][Initialization]

[TODO][Definition (dictionary)]

[TODO][Definition (basis)]

We construct a dictionary as follows (adding auxiliary or slack variables)

The main idea of the simplex algorithm is that of *successive improvements*.
Given one feasible solution we will attempt to find another with a greater value
of the objective function.

We will successively construct new dictionaries.

$$
\begin{align*}
    x_{n + i} &= b_i - &\sum_{j = 1}^n a_{ij} x_j && i = 1, \ldots, m\\
    \hline
    z &= &\sum_{j = 1}^n c_{j} x_j
\end{align*}
$$
(we will always write the objective function as the last last row of the dictionary)
This dictionary is feasible [TODO][Feasible dictionary] if
and only if all the coefficients $b_i$ are nonnegative. We call such problems as problems with *feasible origin*. For the moment we will consider only problems with feasible origin and we will handle initialization for other kinds of problems later.

The variables on left hand side ($x_{n + 1}, \ldots, x_{n+j}) are called *basic* and on the right hand side *nonbasic*.


## Iteration

[TODO][Iteration]

Each iteration consists of
1. Choosing *entering* nonbasic variable
2. Choosing a *leaving* variable
3. Construct next dictionary

[TODO][Pivoting]

- the computational process of constructing a new dictionary

[TODO][Choosing entering variable]

- the choice of entering variable is motivated by desire to improve $z$

**Definition (entering variable)** The *entering variable*
is a *nonbasic variable* $x_j$ with a *positive* coefficient $c_j$ in the last row of the dictionary.

- choose positive coefficient - the rule is ambiguous
    * many candidates
    * no candidate -> optimal value
- there are many rules to choose an entering variable - pivoting rules

[TODO][Choosing a leaving variable]

- non-negativity of the basic variable imposes an upper bound on the increment of the entering variable

**Definition (leaving)** The *leaving variable*
is a *basic variable* whose non-negativity imposes the most stringent upper bound on the increase of the entering variable.

- this rule is ambiguous
    * more than one candidate
    * no candidates (no bounds) -> problem unbounded
- minimum ratio test
- leaving variable is that basic variable that imposes the most stringent bound

[TODO][Degeneracy]

- Consequence of more than one candidate for leaving the basis
- Basic solutions with one or more basic variables at zero are called *degenerate*
- side effect - increase of objective function can be zero
- simplex iterations that do not change the basic solution are called degenerate
- degeneracy is a rule rather than an exception in LP problems
- A situation that forces a degenerate pivot step may occur only for a linear program in which several feasible bases correspond to a single basic feasible solution. Such linear programs are called degenerate.  It is easily seen that in order that a single basic feasible solution be ob-

[TODO][Pivoting]

- exchange of entering and leaving variable

## Termination

[TODO][Termination]

[TODO][Cycling]

It is possible for the simplex algorithm to go through an endless sequence of iterations without finding an optimal solution (if not using the right pivoting rule)

* cycling is the only way to fail Chvatal p.48
* it is possible to avoid cycling - bland rule or perturbation

[TODO][Unboundedness]

## Two-phase simplex

[TODO][two-phase simplex method]

**Phase I**

In Phase I, we apply the revised simplex algorithm to an auxiliary LP problem in order to find an initial feasible solution for the original problem.

**Definition (auxiliary problem)**
$$
\begin{align*}
    \text{minimize}\;&  x_{n+ 1}\\
    \text{subject to}\;&    A x + dx_{n + 1} = b \\
    & x \ge 0
\end{align*}
$$
where $d = -A_B e$ and $e = (1, 1, \ldots, 1)$.

**Phase II**

Solve original problem using the basis found in Phase I.