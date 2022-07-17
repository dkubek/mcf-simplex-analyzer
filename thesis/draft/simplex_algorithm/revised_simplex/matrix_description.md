# Matrix description of a dictionary

For a given basis $(B, N)$ we write the system
$$
Ax = b
$$
as
$$
A_B x_B + A_N x_N = b
$$
explicitly differentiating between basic and nonbasic variables.
Since the matrix $A_B$ is non-singular [Chvatal, geometric representation], we are able to express the basic variables as
$$
x_B = A_B^{-1} b - A_B^{-1} A_N x_N
$$

We do the same with the objective function $z = cx = c_B x_b + c_N x_N$.
Substituting $x_B$ we obtain
$$
z = c_B A_B^{-1} b + (c_N - c_B A_B^{-1} A_N) x_N
$$

The matrix $A_B$ is called the **basic matrix** or the **basis**. We will denote
it by $Q$.