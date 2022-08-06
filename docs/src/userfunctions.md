# User problem
The key to using the `FerriteSolvers.jl` package is to define your 
`problem`. This `problem` should support the following functions
in order for the solver to solve your `problem`.

```@docs
FerriteSolvers.getunknowns
FerriteSolvers.getresidual
FerriteSolvers.getjacobian
FerriteSolvers.update_to_next_step!
FerriteSolvers.update_problem!
FerriteSolvers.calculate_residualnorm
FerriteSolvers.handle_converged!
FerriteSolvers.postprocess!
```

