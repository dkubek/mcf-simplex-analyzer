# Computational form of LP

The simplex algorithm works with a formulation in yet another special form. Additionally we separate the variable bounds

[TODO][Why computational form? --- easier to work with separate variable bounds]

**Definition (Computational form)**
$$
\begin{align*}
    \text{minimize}\;&  c^T x\\
    \text{subject to}\;&    A x = b \\
    & l \le x \le u
\end{align*}
$$
A full row rank, b RHS

### General form to computational form

* Introduce slack variables
* set correct bounds
* NOTE: RHS disappears


**Definition (Computational Standard Form)**
$$
\begin{align*}
    \text{minimize}\;&  c^T x\\
    \text{subject to}\;&    A x = b \\
    & x \ge 0
\end{align*}
$$

The Simplex algorithm was originally developed for problems in standard form (not including individual bounds)