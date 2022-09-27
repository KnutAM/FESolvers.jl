# User problem
The key to using the `FESolvers.jl` package is to define your 
`problem`. This `problem` should support a set of functions
in order for the solver to solve your `problem`. 
While some functions are always required, some are only required by certain solvers. 
Furthermore, a two-level API exist: Simple and advanced. 
The simple API does not expose which solver is used, while the advanced API
requires you to dispatch on the type of solver. 

## Applicable to all solvers
```@docs
FESolvers.getunknowns
FESolvers.getresidual
FESolvers.update_to_next_step!
FESolvers.update_problem!
FESolvers.handle_converged!
FESolvers.postprocess!
```


## Simple API
```@docs
FESolvers.calculate_convergence_measure
FESolvers.getjacobian
FESolvers.getdescentpreconditioner
```

## Advanced API
```@docs
FESolvers.getsystemmatrix
FESolvers.check_convergence_criteria
```

