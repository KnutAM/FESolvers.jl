"""
    getunknowns(problem)

Return the current vector of unknown values
"""
function getunknowns end

"""
    getresidual(problem)

Return the current residual vector of the problem
"""
function getresidual end

""" getjacobian(problem)

Return the jacobian `drdx`, or approximations thereof.

Must be defined for `NewtonSolver`, but can also be 
defined by the advanced API alternative [`getsystemmatrix`](@ref): 
`getsystemmatrix(problem, ::NewtonSolver)`
"""
function getjacobian end

""" getdescentpreconditioner(problem)

Return a preconditioner `K` for calculating the descent direction `p`,
considering solving `r(x)=0` as a minimization problem of `f(x)`
where `r=∇f`. The descent direction is then `p = K⁻¹ ∇f`

Used by the `SteepestDescent` solver, and defaults to `I` if not defined. 
The advanced API alternative is [`getsystemmatrix`](@ref): 
`getsystemmatrix(problem, ::SteepestDescent)`
"""
getdescentpreconditioner(::Any) = LinearAlgebra.I

"""
    getsystemmatrix(problem,nlsolver)

Return the system matrix of the problem. For a Newton solver this
method should return the Jacobian, while for a steepest descent method
this can be a preconditioner as e.g., the L2 product of the gradients.
By default the system matrix for the `SteepestDescent` method is the unity matrix
and thus, renders a vanilla gradient descent solver.
"""
function getsystemmatrix end

"""
    calculate_energy(problem,𝐮)

Return the energy of the system (a scalar) which is the integrated energy
density over the domain Ω.
"""
function calculate_energy end

"""
    update_to_next_step!(problem, time)

Update prescribed values, external loads etc. for the given time.

This function is called in the beginning of each new time step. 
Note that for adaptive time stepping, it may be called with a lower 
time than the previous time if the solution did not converge.
"""
function update_to_next_step! end

"""
    update_problem!(problem, Δx, update_spec::UpdateSpec)

Update the unknowns of the problem by `Δx` according to `update_spec`.
Note that 
- Some linear solvers may be inaccurate, and if a modified stiffness is used 
  to enforce constraints on `x`, it is good the force `Δx=0` on these
  components inside this function. 
- `Δx=nothing` in the first call after [`update_to_next_step!`](@ref)
  in which case, typically, no change of `x` should be made. Dirichlet
  boundary conditions are typically updated in `update_to_next_step!`.

The `update_spec` gives the information about what and how to update.
See the documentation for [`UpdateSpec`](@ref) for further details. 
This feature is used by some nonlinear solvers to customize the iteration 
strategy to speed up or aid convergence. For basic cases when getting started, 
this can be ignored and a simple function definition would be 
```julia
FESolvers.update_problem!(problem, Δx, _)
```


    update_problem!(problem, Δx; update_residual::Bool, update_jacobian::Bool)

The old but now deprecated interface is still available without `update_spec`.
The instructions are here:

* Assemble the residual if `update_residual=true`
* Assemble the jacobian if `update_jacobian=true`

"""
function update_problem! end


"""
    calculate_convergence_measure(problem, Δa, iter) -> AbstractFloat

Calculate a value to be compared with the tolerance of the nonlinear solver. 
A standard case when using [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
is `norm(getresidual(problem)[Ferrite.free_dofs(ch)])` 
where `ch::Ferrite.ConstraintHandler`. `Δa` is the update of the unknowns from 
the previous iteration. Note that `iter=1` implies `Δa=0`
"""
function calculate_convergence_measure end

"""
    postprocess!(problem, solver)

Perform any postprocessing at the current time.
Called at the beginning of the simulation, and directly
after time step converged (right before `handle_converged!`).
"""
function postprocess! end

# Deprecated
function postprocess!(problem, solver)
    @warn "`postprocess!(problem, step, solver)` is deprecated, overload `postprocess!(problem, solver)` instead" maxlog=1
    return postprocess!(problem, get_step(solver), solver)
end

"""
    handle_converged!(problem)

Do necessary update operations once it is known that the 
problem has converged. E.g., update old values to the current. 
Only called after the problem has converged, after `postprocess!`
"""
function handle_converged! end

"""
    handle_notconverged!(problem, solver)

Optional function to make changes to the problem in case it did not converge.
If not implemented, this defaults to a no-op.
"""
handle_notconverged!(::Any, ::Any) = nothing

"""
    setunknowns!(problem, x)

Copy the given values `x` into the unknown values of `problem`. 
Defaults to `copy!(getunknowns(problem), x)`, which works as long as
`getunknowns` returns the `Vector{<:Number}` stored in the problem struct. 
If, e.g. the unknowns is a custom type or a nested vector, this function should 
be overloaded. 
"""
setunknowns!(problem, x::Vector{<:Number}) = copy!(getunknowns(problem), x) 

"""
    close_problem(problem)

This function is called after solving the problem, even if the solution 
fails due to an error thrown, for example if the problem doesn't converge. 
Use to close any file streams etc. that are open and should be closed
"""
close_problem(::Any) = nothing
