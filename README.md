# FerriteSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KnutAM.github.io/FerriteSolvers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KnutAM.github.io/FerriteSolvers.jl/dev)
[![Build Status](https://github.com/KnutAM/FerriteSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KnutAM/FerriteSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KnutAM/FerriteSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KnutAM/FerriteSolvers.jl)

Package that provides interfaces to different solvers, with three key areas: 

* Linear solvers: Solving ``Ax=b``
* Nonlinear solvers: Solving ``r(x)=0`` 
* Time stepping

The function 
```julia
solve_ferrite_problem!(solver::FerriteSolver, problem)
```
is given, where `problem` is a user specified datatype, 
for which the following functions should be defined

```julia
a = getunknowns(problem)
update_to_next_step!(problem, t) # Update boundary conditions etc. for a new time step
update_problem!(problem, Δa) # Assemble stiffness and residual for a+=Δa, after ensuring that Δa is zero at Dirichlet BC. 
converged = isconverged(problem, tolerance) # Check if the problem has converged within the given tolerance
r = getresidual(problem)
K = getjacobian(problem)    # K = dr/da
handle_converged!(problem, t)   # Do stuff if required after the current time step has converged. 
postprocess(problem, i) # Do all postprocessing for step i at time t
```
