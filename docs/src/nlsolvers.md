# Nonlinear solvers
The following nonlinear solves are included
```@docs
NewtonSolver
```


## Implementation of custom nonlinear solvers
A nonlinear solver should support the `solve_nonlinear!` function specified below. 

```@docs
FerriteSolvers.solve_nonlinear!
```