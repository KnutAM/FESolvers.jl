# FerriteSolvers
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/FerriteSolvers.jl/dev)
[![Build Status](https://github.com/KnutAM/FerriteSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KnutAM/FerriteSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KnutAM/FerriteSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KnutAM/FerriteSolvers.jl)

Package to easily solve nonlinear problem, in particularily tailored to [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl).
By defining `solver = FerriteSolver(nlsolver, timestepper)`, the function 
```julia
solve_ferrite_problem!(solver::FerriteSolver, problem)
```
solves the user-defined `problem`. For this user defined type, 
the following functions should be defined

```julia
x = getunknowns(problem)
update_to_next_step!(problem, t)# Update boundary conditions etc. for a new time step
update_problem!(problem, Δx)    # Assemble stiffness and residual for x+=Δx 
calculate_residualnorm(problem) # Get scalar value of residual
r = getresidual(problem)
K = getjacobian(problem)        # K = dr/dx
handle_converged!(problem, t)   # Do stuff if required after the current time step has converged. 
postprocess!(problem, step)     # Do all postprocessing for current step (after convergence)
```
