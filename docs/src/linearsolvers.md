# Linear solvers
The following linear solves are included
```@docs
BackslashSolver
```

## Implementation of custom linear solvers
A linear solver should support the `update_guess!` function specified below. 

```@docs
FerriteSolvers.update_guess!
```
