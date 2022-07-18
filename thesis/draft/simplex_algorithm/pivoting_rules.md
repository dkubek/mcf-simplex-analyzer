# Pivoting rules

Here we describe the rules for choosing an entering variable based on the current form of the objective function. A pivot rule is a rule for chosing an entering variable if there are multiple possible choices.

The number of iterations needed for solving a linear program depends heavily on a pivot rule.

[TODO][Dantzig]

**Definition (Dantzig rule, Largest Coefficient)**
Choose a non-basic variable with largest coefficient.

[TODO][Bland]

**Definition (Bland)**
Choose the non-basic entering variable with the smallest index and also the leaving variable with smallest index. This rule prevents cycling.

[TODO][Largest increase]

**Definition (Largest Increase)**
Choose a non-basic variable with largest absolute improvement in $z$.

[TODO][Steepest edge]

- best in practice
- Devex

[TODO][Random]

**Definition (Random edge rule)**
Select the entering variable uniformly at random.