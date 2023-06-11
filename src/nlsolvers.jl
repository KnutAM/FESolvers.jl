"""
    UpdateSpec(;jacobian, residual, type=nothing)

An `UpdateSpec` is sent to `update_problem!` to pass the problem
information about how it should be updated. The following methods 
should be used by the problem to request information

* `should_update_jacobian(::UpdateSpec)::Bool`: Self explanatory
* `should_update_residual(::UpdateSpec)::Bool`: Self explanatory
* `get_update_type(::UpdateSpec)`: How should the problem be updated?
  This is used by special solvers where, for example, different 
  approximations of the stiffness is available, see e.g. 
  [`AdaptiveNewtonSolver`](@ref)
"""
struct UpdateSpec{JT}
    jacobian::Bool      # Should the jacobian be updated?
    residual::Bool      # Should the residual be updated?
    type::JT            # What type of update should be performed?
end
function UpdateSpec(;jacobian, residual, type=nothing)
    return UpdateSpec(jacobian, residual, type)
end
should_update_jacobian(us::UpdateSpec) = us.jacobian
should_update_residual(us::UpdateSpec) = us.residual
get_update_type(us::UpdateSpec) = us.type

# Temporary solution to support old interface
# TODO: Remove after some time...
function update_problem!(problem, Δx, update_spec#=::UpdateSpec=#)
    return update_problem!(problem, Δx; 
        update_jacobian = should_update_jacobian(update_spec), 
        update_residual = should_update_residual(update_spec)
        )
end

"""
    getmaxiter(nlsolver)

Returns the maximum number of iterations allowed for the nonlinear solver
"""
function getmaxiter end

"""
    getnumiter(nlsolver)

Returns the last number of iterations used by the nonlinear solver
"""
function getnumiter end

"""
    get_convergence_measures(nlsolver[, i])

Get the convergence measures for iteration `i` from the nlsolver. 
If `i` is not given, the full history is returned. 
Note that unless `i::Int` is given, a new vector will be allocated.
"""
function get_convergence_measures end

"""
    gettolerance(nlsolver)

Returns the iteration tolerance for the solver
"""
function gettolerance end

"""
    update_state!(nlsolver, problem, r)

A nonlinear solver may solve information about its convergence state.
`r` is the output from [`calculate_convergence_measure`](@ref) when 
this function is called by the default implementation of 
[`check_convergence_criteria`](@ref). 
"""
update_state!(_, _, _) = nothing

"""
    reset_state!(nlsolver)

This function is used to reset `nlsolver`'s state at the 
beginning of each new time step. 
"""
reset_state!(::Any) = nothing

"""
    get_initial_update_spec(nlsolver)::UpdateSpec

How to update initially (before starting time stepping)
"""
function get_initial_update_spec end

"""
    do_initial_update(nlsolver)

Should the problem be updated initially, before starting time stepping?
Normally, not required to overload as the update specification from 
`get_initial_update_spec` is used to decide how if an update is required. 
"""
function do_initial_update(nlsolver)
    us = get_initial_update_spec(nlsolver)
    return should_update_jacobian(us) || should_update_residual(us)
end

"""
    get_first_update_spec(nlsolver, last_converged::Bool)

Get the update specification for `nlsolver` in the first iteration of the time step.
`last_converged` indicates if the previous time step converged, or if this is a retry 
using, e.g., a different time step. 
"""
function get_first_update_spec end

"""
    get_update_spec(nlsolver)::UpdateSpec

Get the update specification during regular iterations. It is the `nlsolver`'s job 
to keep track of any state required for deciding changes to the update specification 
during iterations. 
"""
function get_update_spec end

# Optional methods 
"""
    get_linesearch(nlsolver)
Returns the used linesearch of the nonlinear solver.
"""
function get_linesearch end 

"""
    get_linear_solver(nlsolver)

Get the linear solver used by the nonlinear solver
"""
function get_linear_solver end

"""
    solve_nonlinear!(problem, nlsolver, last_converged)

Solve the current time step in the nonlinear `problem`, (`r(x) = 0`),
by using the nonlinear solver `nlsolver`. `last_converged::Bool` 
is just for information if the last time step converged or not. 
In many cases it suffices to overload [`calculate_update!`](@ref) 
for a custom nonlinear solver. 
"""
function solve_nonlinear!(problem, nlsolver, last_converged)
    maxiter = getmaxiter(nlsolver)
    reset_state!(nlsolver)
    update_problem!(problem, nothing, get_first_update_spec(nlsolver, last_converged))
    Δa = zero(getunknowns(problem))
    for iter in 1:maxiter
        check_convergence_criteria(problem, nlsolver, Δa, iter) && return true
        
        maybe_reset_problem!(problem, Δa, nlsolver) # No-op for standard nlsolver, but can be customized
        
        calculate_update!(Δa, problem, nlsolver)          # Newton's method: Δa .= -K\r 
        update_problem!(problem, Δa, get_update_spec(nlsolver)) # a += Δa and update K and r according to update_spec
    end
    check_convergence_criteria(problem, nlsolver, Δa, maxiter+1) && return true
    return false
end

"""
    function calculate_update!(Δx, problem, nlsolver)

According to the nonlinear solver, `nlsolver`, at iteration `iter`,
calculate the update, `Δx` to the unknowns `x`.
"""
function calculate_update!(Δa, problem, nlsolver)
    r = getresidual(problem)
    K = getsystemmatrix(problem, nlsolver)
    solve_linear!(Δa, K, r, get_linear_solver(nlsolver))
    linesearch!(Δa, problem, get_linesearch(nlsolver)) # Scale Δa
    return Δa
end

"""
    maybe_reset_problem!(problem, Δa_old, nlsolver)

*Note:* Custom nonlinear solvers may rely on the no-op default implementation.

After checking for convergence, the default implementation of `solve_nonlinear`
will call this function, allowing the nlsolver to call `update_problem!(problem, -Δa)`
to reset the problem to the state at the last iteration (this is useful if e.g. the 
jacobian should be calculated differently, or if a higher dampening factor should be 
used when calculating the update).
"""
maybe_reset_problem!(problem, Δa, nlsolver) = nothing 
