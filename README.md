# FESolvers
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/FESolvers.jl/dev)
[![Build Status](https://github.com/KnutAM/FESolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KnutAM/FESolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KnutAM/FESolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KnutAM/FESolvers.jl)

Package to easily solve nonlinear problem, in particularily tailored to [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl).

## Your problem - your way
You define your own problem type, `FESolvers` 
just needs to be able to request updates and access to variables. 

For a standard case, using Newton iterations to solve the `problem`,
$$\mathbf{r}(\mathbf{x}(t),t) = \mathbf{0}$$
the following `get*` functions
```julia
x = getunknowns(problem)
r = getresidual(problem)
K = getjacobian(problem)    # ∂r/∂x
```
must be defined.

Define the solver and timestepper, e.g. 
```julia
solver = QuasiStaticSolver(;nlsolver=NewtonSolver(), timestepper=FixedTimeStepper(collect(0:0.1:1.0)))
```

The problem can then be solved by calling 
```julia
solve_problem!(problem, solver)
```

For each new time, `t`, in `timestepper`, the user-defined function 
```julia
update_to_next_step!(problem, t)
```
is called to update boundary conditions and other time-dependent items to `t`.
Thereafter, the user-defined function
```julia 
update_problem!(problem, nothing; update_residual::Bool, update_jacobian::Bool)
```
is called. If requested by the boolean kwargs, the corresponding items should be updated 
(Newton's method always requests these to be updated) following the updated boundary conditions. 
Then starts the iterations to find the new value of the unknowns, `x`.
### Iterations
First, the residual is checked by calling the user-defined function
```julia
calculate_convergence_measure(problem, Δx, iter) < gettolerance(nlsolver)
```
where the user defines `calculate_convergence_measure`. If `false`, then 
continue by calculating the update, `Δx=-[∂r/∂x]\r` using
the linear solver given to `nlsolver`. 

Then, call once more 
```julia 
update_problem!(problem, Δx; update_residual::Bool, update_jacobian::Bool)
```
but this time with the update `Δx` such that the unknowns, `x`, the residual, 
and the jacobian can be updated. Keep looping over the iterations until
`calculate_convergence_measure` returns a sufficiently small value.

### After convergence
The first thing that happens after convergence is that the user-defined function 
```julia
postprocess!(problem, step, [solver])   # Optional to define with or without solver
```
is called. This can be used to save any data required for the given step.
Directly after, 
```julia
handle_converged!(problem)
```
is called. This should be used to update old values, such as 
old unknowns, old time, old state variables etc. to the current values. 

### End of simulation
At the end of the simulation, even if it fails due to an error, 
the function
```julia
close_problem(problem)
``` 
is called. This allows closing any open files etc. 

## Summary of functions to define in standard cases
```julia
x = getunknowns(problem)
r = getresidual(problem)
K = getjacobian(problem)                # K = ∂r/∂x
update_to_next_step!(problem, t)        # Update boundary conditions etc. for a new time step
update_problem!(problem, Δx; kwargs...) # Assemble stiffness and residual for x+=Δx 
calculate_convergence_criterion(problem)# Get a scalar value to compare with the iteration tolerance
postprocess!(problem, step)             # Do all postprocessing for current step (after convergence)
handle_converged!(problem, t)           # Do stuff if required after the current time step has converged. 
close_problem(problem)                  # Close any open file streams etc. Called in a `finally` block. 
```
