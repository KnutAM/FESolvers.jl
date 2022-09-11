# Nonlinear solvers
A nonlinear solver should support the `solve_nonlinear!` function specified below. 

```@docs
FerriteSolvers.solve_nonlinear!
```

It can do so, by supporting the following functions
```@docs
FerriteSolvers.update_unknowns!
FerriteSolvers.getmaxiter
FerriteSolvers.gettolerance
```
and optionally
```@docs
FerriteSolvers.update_state!
FerriteSolvers.reset_state!
```

## Implemented Solvers

```@docs
NewtonSolver
SteepestDescent
```

### Linesearch
Some nonlinear solvers can use linesearch as a compliment, 
and the following linesearches are included. 
```@docs
NoLineSearch
ArmijoGoldstein
```