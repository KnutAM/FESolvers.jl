# User problem
The key to using the `FerriteSolvers.jl` package is to define your 
`problem`. This `problem` should support a set of functions
in order for the solver to solve your `problem`. 
While some functions are always required, some are only required by certain solvers. 
Furthermore, a two-level API exist: Simple and advanced. 
The simple API does not expose which solver is used, while the advanced API
requires you to define different methods depending on the type of solver. 

## Applicable to all solvers
```@docs
FerriteSolvers.getunknowns
FerriteSolvers.getresidual
FerriteSolvers.update_to_next_step!
FerriteSolvers.update_problem!
FerriteSolvers.handle_converged!
FerriteSolvers.postprocess!
```


## Simple API
```@docs
FerriteSolvers.calculate_convergence_measure
FerriteSolvers.getjacobian
FerriteSolvers.getdescentpreconditioner
```

## Advanced API
```@docs
FerriteSolvers.getsystemmatrix
FerriteSolvers.check_convergence_criteria
```

