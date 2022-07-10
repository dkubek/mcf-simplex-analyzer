# How can we solve LP problems?

## Simplex algorithm

In 1947 G. B. Dantzig designed the “simplex method” for solving linear
programming formulations of U.S. Air Force planning problems.  It has been added
to the list as one of the top ten algorithms with the he greatest influence on
the development and practice of science  and engineering in the 20th century
[10.1109/MCISE.2000.814652]. 

The simplex algorithm begins with a primal feasible basis and uses pricing
operations until an optimum solution is computed. It searches for an optimal
solution by moving from one adjacent feasible solution to another, along the
edges of the feasible region. It also guarantees monotonicity of the objective
value. 

Unfortunately, the worst case complexity of the simplex method is exponential as
shown by [HOW GOOD IS THE SIMPLEX ALGORITHM - Klee & Minty]. However, on real
world problems the simplex behaves surprisingly well and the number of
iterations scales linearly with the number of constraints.

Many efforts have been made since Dantzig's initial proposed algorithm order to
enhance the performance.

## Ellipsoid method

In 1978 Khachiyan proposed the first exact polynomial algorithm for LP
[Polynomial algorithms in linear programming, Khachian], the so called ellipsoid
method.

The main impact of the ellipsoid method had a great impact on the theory of LP,
but the algorithm cannot compete with the simplex algorithm in practice due to
the expensive execution time per iteration.

## Interior Point Methods

Other researchers proposed the interior point algorithms that traverse across
the interior of the feasible region [TODO:what is feasible region]. However, due
to the expensive execution time per iteration and possible numerical
instability, the interior point methods did not compete very well in practice
with the simplex algorithm.

In 1984 the first interior point method that outperformed simplex algorithm was
proposed by Karmarkar [ E W POLYNOMIAL-TIME ALGORITHM FOR LINEAR PROGRAMMING].
Many improvements have been made in theory and in practice of interior
point methods since.

## State of the art

Nowadays, large-scale LPs can be solved either by a primal simplex algorithm or
a dual simplex algorithm (or a combination of a primal and a dual algorithm) or
an interior point algorithm. 

Interior point methods and simplex algorithm continue to be valuable in practice.

[TODO] Find newert souce of state-of-the-art