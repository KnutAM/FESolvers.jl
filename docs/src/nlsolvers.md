# Nonlinear solvers
A nonlinear solver should support the `solve_nonlinear!` function specified below. 

```@docs
FESolvers.solve_nonlinear!
```

It can do so, by supporting the following functions
```@docs
FESolvers.calculate_update!
FESolvers.getmaxiter
FESolvers.gettolerance
```
and optionally
```@docs
FESolvers.update_state!
FESolvers.reset_state!
```

The implemented solvers additionally support 
```@docs
FESolvers.getnumiter
FESolvers.getlinesearch
```

## Implemented Solvers

```@docs
LinearProblemSolver
NewtonSolver
SteepestDescent
```

### Linesearch
Some nonlinear solvers can use linesearch as a complement, 
and the following linesearches are included. 
```@docs
NoLineSearch
ArmijoGoldstein
```