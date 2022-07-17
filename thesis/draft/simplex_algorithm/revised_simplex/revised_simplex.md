*[How does Revised Simplex work?]*

**Step 0** (Initialization).

*Presolve the LP problem.*

*Scale the LP problem.*

Select an initial basic solution (B, N).
If the initial basic solution is feasible then proceed to step 2.

**Step 1** (Phase I).

Construct an auxiliary problem 

Apply the revised simplex algorithm. If the final basic solution (B, N) is
feasible, then proceed to step 2 in order to solve the initial problem. The LP
problem can be either optimal or infeasible.

**Step 2** (Phase II).

**Step 2.1** (Test of Optimality).

Choose the index $j$ of the entering variable using a pivoting rule. Variable $x_j$ enters the basis.

**Step 2.2**(Pivoting).

Compute the pivot column $d$

Choose the leaving variable

**Step 2.3** (Update).

Swap indices k and l. Update the new basis inverse AB , using a basis update
scheme. Update vectors $xB$ , $w$, and $s_N$ .

[FIGURE: Revised simplex algorithm flowchart]


## Choosing the entering variable

In the standard simplex, the objective function was given in the last row of our dictionary and was readily available. To chose an entering variable all we had to do
was chose a variable with a positive coefficient (according to some rule). However,
in the revised simplex method the objective function is given as $c_N - c_B^T Q^{-1} A_N$

We compute this vector by first finding $y = c_B Q$ by solving the system $y^T Q = c_B$ and then we calculate $c_N - y^T A_N$.

We then choose the entering variable as in standard.

Note that it is possible that individual components of $c_N - y^T A_N$ may be calculated indiviadually. For $x_j$ nonbasic variable corresponding to $c_j$ of $c_N$ and column $a$ of $A_N$ the relevant component equals $c_J - ya$. 
The entering variable may be any nonbasic variable $x_j$ for which $ya < c_j$.
This fact is used in many efficient implementations.

## Choosing the leaving variable

To determine the leaving variable we increase the value $t$ of the entering variable
from zero to some positive level maintaining the values of the remaining nonbasic variables at their zero levels and adjusting the values of the basic variables so as to preserve $Ax = b$. As $t$ increases the value of of the basic variable change until one hits the zero and this one is the leaving variable. Therefore we have to know the largest possible value of $t$.

In standard simplex we did this by ratio test with information readily available in the dictionary.

In revised simplex we have
$$
    x_B  = x_B^* + - Q^{-1} A_N x_N
$$
and so $x_B$ changes from $x_B^*$ to $x_B^* - t d$ where $d$ is a column of $B^-1 A_N$ corresponding to the entering variable. (Note $d = B^-1 a$, $a$ is the entering column)
We find the largest $t$ such that $x_B^* - td \ge 0$, if there is no such $t$, the problem is unbounded.

To obtain $d$ we have to solve $Bd = a$.

## Was this worth it?

So far it was required of us to do computations not required in the standard simplex
method. We might ask whether this was worth it or not. The main speedup comes from the end of the iteration. Where standard simplex has to do a laborious update of the whole dictionary, no such computation is needed in the revised simplex method.J

## Update the basis

$B := B \ \{leaving\} \cup \{entering\}$


*[TODO: Revised simplex vs standard simplex]*

On large sparse problems encountered in practice, an iteration of the revised simplex method takes less time than an iteration of the standard simplex method.
The revised simplex method beats the standard simplex method as soon as the number of rows exceeds ~100. [Chvatal]