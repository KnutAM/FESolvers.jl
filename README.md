# FerriteSolvers
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/FerriteSolvers.jl/dev)
[![Build Status](https://github.com/KnutAM/FerriteSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KnutAM/FerriteSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KnutAM/FerriteSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KnutAM/FerriteSolvers.jl)

Package to easily solve nonlinear problem, in particularily tailored to [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl).
The main goal is a flexible solution scheme, allowing extensible and modular solver parts, divided into the following three 
categories areas

* Nonlinear solvers: Solving $\boldsymbol{r}(\boldsymbol{x})=\boldsymbol{0}$
* Time stepping
* Linear solvers: Solving $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$

through the `solver = FerriteSolver(nlsolver, timestepper, linearsolver)`

The function 
```julia
solve_ferrite_problem!(solver::FerriteSolver, problem)
```
is exported, where `problem` is a user specified datatype, 
for which the following functions should be defined

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
