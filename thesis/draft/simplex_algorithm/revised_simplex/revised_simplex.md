*[How does Revised Simplex work?]*

**Step 0** (Initialization).

Presolve the LP problem.

Scale the LP problem.

Select an initial basic solution (B, N).
If the initial basic solution is feasible then proceed to step 2.

**Step 1** (Phase I).

Construct an auxiliary problem 

Apply the revised simplex algorithm. If the final basic solution (B, N) is
feasible, then proceed to step 2 in order to solve the initial problem. The LP
problem can be either optimal or infeasible.

**Step 2** (Phase II).

**Step 2.1** (Test of Optimality).

Choose the index l of the entering variable using a pivoting rule. Variable x_l enters the basis.

**Step 2.2**(Pivoting).

Compute the pivot column $h_l$

Choose the leaving variable

**Step 2.3** (Update).

Swap indices k and l. Update the new basis inverse AB , using a basis update
scheme. Update vectors $xB$ , $w$, and $s_N$ .

[FIGURE: Revised simplex algorithm flowchart]


*[TODO: Revised simplex vs standard simplex]*
