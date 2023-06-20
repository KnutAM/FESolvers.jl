# =========================================================================== #
# UpdateSpec
# =========================================================================== #
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

# =========================================================================== #
# SolverState
# =========================================================================== #
mutable struct SolverState{T,C}
    numiter::Int
    const residuals::Vector{T}
    status::C
    function SolverState(residuals::Vector{T}, cs::C) where {T,C}
        return new{T,C}(0, residuals, cs)
    end
end
function SolverState(maxiter::Int)
    return SolverState(map(Returns(NaN), 1:(maxiter+1)), ConvergenceStatus(true))
end

get_num_iter(sh::SolverState) = sh.numiter

function reset_solver_state!(sh::SolverState)
    sh.numiter = 0
    fill!(sh.residuals, NaN)
    sh.status = ConvergenceStatus(false, :nonlinearsolver)
end

function update_solver_state!(sh::SolverState, nlsolver, residual)
    sh.numiter += 1
    sh.residuals[sh.numiter] = residual
    if residual < get_tolerance(nlsolver)
        sh.status = ConvergenceStatus(true)
    end
end

is_converged(sh::SolverState) = is_converged(sh.status)
check_convergence(sh::SolverState) = check_convergence(sh.status)
is_failed(sh::SolverState) = isa(sh.status.exception, Exception)

function set_failure!(sh::SolverState, reason, exception)
    sh.status = ConvergenceStatus(false, reason, exception)
end

# See docstring for get_convergence_measure(nlsolver)
get_convergence_measure(sh::SolverState) = sh.residuals[sh.numiter]
function get_convergence_measure(sh::SolverState, ind::Integer)
    checkbounds(1:get_num_iter(sh), ind)
    return sh.residuals[ind]
end
function get_convergence_measure(sh::SolverState, inds)
    checkbounds(1:get_num_iter(sh), inds)
    return view(sh.residuals, inds)
end
get_convergence_measure(sh::SolverState, ::Colon) = view(sh.residuals, 1:sh.numiter)

# =========================================================================== #
# General nonlinear solver interface
# =========================================================================== #¨

# Functions that normally don't require overloading if `SolverState` is included,
# and `get_solver_state` is implemented. 
"""
    is_converged(nlsolver)

Has the `nlsolver` converged? Reset by [`reset_nlsolver!`](@ref).
"""
is_converged(nlsolver) = is_converged(get_solver_state(nlsolver))

check_convergence(nlsolver) = check_convergence(get_solver_state(nlsolver))

"""
    get_num_iter(nlsolver)

Returns the last number of iterations used by the nonlinear solver
"""
get_num_iter(nlsolver) = get_num_iter(get_solver_state(nlsolver))

"""
    get_convergence_measure(nlsolver)

Get the last convergence measure for nlsolver. 

    get_convergence_measure(nlsolver, k::Integer)
    
Get the `k`th convergence measure in the current iteration
    
    get_convergence_measure(nlsolver, inds)

Get a view to the vector of the convergence measures for 
iterations `inds`, i.e. `view(residuals, inds)`

    get_convergence_measure(nlsolver, ::Colon)

Get a view to the vector of all convergence measures, i.e. 
`get_convergence_measure(nlsolver, 1:get_num_iter(nlsolver))`
"""
get_convergence_measure(nlsolver, args...) = get_convergence_measure(get_solver_state(nlsolver), args...)

"""
    reset_solver_state!(nlsolver)

Called at the beginning of each new time step, and resets the 
solver's status and potentially convergence history etc. 
"""
function reset_solver_state!(nlsolver)
    reset_solver_state!(get_solver_state(nlsolver))
end

"""
    update_solver_state!(nlsolver, problem, r)

A nonlinear solver may solve information about its convergence state.
`r` is the output from [`calculate_convergence_measure`](@ref) when 
this function is called by the default implementation of 
[`check_convergence_criteria`](@ref). 
"""
function update_solver_state!(nlsolver, _, r)
    update_solver_state!(get_solver_state(nlsolver), nlsolver, r)
end

set_failure!(nlsolver, args...) = set_failure!(get_solver_state(nlsolver), args...)
is_failed(nlsolver) = is_failed(get_solver_state(nlsolver))

# Functions that always should be implemented
"""
    get_solver_state(nlsolver)

All nonlinear solvers that contain a `SolverState` should normally overload 
this function to make many other functions work automatically. 
"""
function get_solver_state end

"""
    get_max_iter(nlsolver)

Returns the maximum number of iterations allowed for the nonlinear solver
"""
function get_max_iter end

"""
    get_tolerance(nlsolver)

Returns the iteration tolerance for the solver
"""
function get_tolerance end

"""
    get_initial_update_spec(nlsolver)::UpdateSpec

How to update initially (before starting time stepping)
"""
function get_initial_update_spec end

"""
    get_first_update_spec(nlsolver)::UpdateSpec

Get the update specification for `nlsolver` in the first iteration of the time step.
"""
function get_first_update_spec end

"""
    get_update_spec(nlsolver)::UpdateSpec

Get the update specification during regular iterations. It is the `nlsolver`'s job 
to keep track of any state required for deciding changes to the update specification 
during iterations. 
"""
function get_update_spec end

# Update spec function that normally don't require custom implementation
"""
    should_do_initial_update(nlsolver)

Should the problem be updated initially, before starting time stepping?
Normally, not required to overload as the update specification from 
`get_initial_update_spec` is used to decide how if an update is required. 
"""
function should_do_initial_update(nlsolver)
    us = get_initial_update_spec(nlsolver)
    return should_update_jacobian(us) || should_update_residual(us)
end

# Interface for adaptive nonlinear solvers (defaults for other cases)
"""
    should_reset_problem(nlsolver)

*Note:* Custom nonlinear solvers may rely on the default `false` return value.

If this function returns true, the problem will be reset as
`update_problem!(problem, -Δa)` to reset the problem to the state 
at the last iteration (e.g. when switching the jacobian calculation)
"""
should_reset_problem(nlsolver) = false 

# Methods that are used in calculate_update! if not overloaded
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

# Top-level method to overload
"""
    solve_nonlinear!(problem, nlsolver, last_converged)

Solve the current time step in the nonlinear `problem`, (`r(x) = 0`),
by using the nonlinear solver `nlsolver`. `last_converged::Bool` 
is just for information if the last time step converged or not. 
In many cases it suffices to overload [`calculate_update!`](@ref) 
for a custom nonlinear solver. 
"""
function solve_nonlinear!(problem, nlsolver)
    maxiter = get_max_iter(nlsolver)
    update_problem!(problem, nothing, get_first_update_spec(nlsolver))
    Δa = zero(getunknowns(problem))
    reset_solver_state!(nlsolver)
    for iter in 1:maxiter
        # Check convergence and update solver state
        r = calculate_convergence_measure(problem, Δa, iter)
        update_solver_state!(nlsolver, problem, r)
        is_converged(nlsolver) && return nothing

        # Check if we should reset the problem (for adaptive solvers)
        if should_reset_problem(nlsolver)
            reset_problem!(problem, nlsolver; Δx_old=Δa)
        end
        
        # Update the guess for the solution
        calculate_update!(Δa, problem, nlsolver)          # Newton's method: Δa .= -K\r 
        update_problem!(problem, Δa, get_update_spec(nlsolver)) # a += Δa and update K and r according to update_spec
        is_failed(nlsolver) && return nothing
    end
    r = calculate_convergence_measure(problem, Δa, maxiter+1)
    update_solver_state!(nlsolver, problem, r)
    return nothing
end

# Alternative method to overload if solve_nonlinear! can be used. 
"""
    function calculate_update!(Δx, problem, nlsolver)

According to the nonlinear solver, `nlsolver`, at iteration `iter`,
calculate the update, `Δx` to the unknowns `x`.
"""
function calculate_update!(Δa, problem, nlsolver)
    r = getresidual(problem)
    K = getsystemmatrix(problem, nlsolver)
    try
        solve_linear!(Δa, K, r, get_linear_solver(nlsolver))
    catch e
        if isa(e, LinearAlgebra.SingularException)
            set_failure!(nlsolver, :linearsolver, e)
        else
            throw(e)
        end
    end
    
    linesearch!(Δa, problem, get_linesearch(nlsolver)) # Scale Δa
    return Δa
end

# This method should not be necessary to overload!
"""
    reset_problem!(problem, nlsolver; x, Δx_old)

Reset the problem, either by giving the new vector of unknown values, `x`,
**or** the last increment to be reset, `Δx_old`. `x` will **not** be modified
if given, but `Δx_old` will be **modified** to `-Δx_old` if given. 
"""
function reset_problem!(problem, nlsolver; x=nothing, Δx_old=nothing)
    current_x = getunknowns(problem)
    calc_Δx(::Nothing, ::Nothing, _) = error("x or Δx must be given")
    function calc_Δx(::AbstractVector, ::AbstractVector, _)
        error("Either x or Δx must be given (not both)")
    end
    calc_Δx(::Nothing, Δx_old::AbstractVector, _) = map!(-, Δx_old, Δx_old)
    calc_Δx(x::AbstractVector, ::Nothing, current_x) = x .- current_x
    Δx = calc_Δx(x, Δx_old, current_x)
    update_problem!(problem, Δx, get_update_spec(nlsolver))
end
