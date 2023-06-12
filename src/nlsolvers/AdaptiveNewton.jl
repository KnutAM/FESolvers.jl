"""
    AdaptiveNewtonSolver(;
        update_types, switch_criterion,
        linsolver=BackslashSolver(), linesearch=NoLineSearch(),
        maxiter=10, tolerance=1e-6, update_jac_first=true)

Define an adaptive newton solver, where the update type 
(given to [`UpdateSpec`](@ref FESolvers.UpdateSpec))
changes during the iterations. The different options 
are given to `update_types`, and the criterion to switch 
between them is given to `switch_criterion`. Remaining options
are similar to the standard newton solver. 
"""
Base.@kwdef mutable struct AdaptiveNewtonSolver{LinearSolver,LineSearch,T,UT,SC}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = NoLineSearch()
    const maxiter::Int = 10
    const tolerance::T = 1e-6
    const update_jac_first::Bool=true # Should jacobian be updated first iteration (or use the one from the previous converged)
    const update_types::UT
    update_type_nr::Int=1
    const switch_criterion::SC
    reset_problem::Bool=false
    numiter::Int = 0  # Current number of iterations (reset at beginning of new iteration)
    const residuals::Vector{T} = zeros(typeof(tolerance),maxiter+1)  # Last step residual history
end
getsystemmatrix(problem, ::AdaptiveNewtonSolver) = getjacobian(problem)

_get_update_type(nls::AdaptiveNewtonSolver) = nls.update_types[nls.update_type_nr]
function get_initial_update_spec(nls::AdaptiveNewtonSolver)
    return UpdateSpec(;jacobian=!(nls.update_jac_first), residual=false, type=_get_update_type(nls))
end
function get_first_update_spec(nls::AdaptiveNewtonSolver, last_converged::Bool)
    if last_converged
        return UpdateSpec(jacobian=nls.update_jac_first, residual=true, type=_get_update_type(nls))
    else # We must update if jacobian was updated for each
        return UpdateSpec(jacobian=nls.update_jac_each, residual=true, type=_get_update_type(nls))
    end
end
function get_update_spec(nls::AdaptiveNewtonSolver)
    return UpdateSpec(;jacobian=true, residual=true, type=_get_update_type(nls))
end

get_linesearch(nlsolver::AdaptiveNewtonSolver) = nlsolver.linesearch
get_linear_solver(nlsolver::AdaptiveNewtonSolver) = nlsolver.linsolver

getmaxiter(nlsolver::AdaptiveNewtonSolver) = nlsolver.maxiter
gettolerance(nlsolver::AdaptiveNewtonSolver) = nlsolver.tolerance
getnumiter(s::AdaptiveNewtonSolver) = s.numiter
get_convergence_measures(s::AdaptiveNewtonSolver, inds=1:getnumiter(s)) = s.residuals[inds]

get_type_number(s::AdaptiveNewtonSolver) = s.update_type_nr

function reset_state!(s::AdaptiveNewtonSolver)
    s.numiter = 0
    fill!(s.residuals, 0)
    s.update_type_nr = 1
    s.reset_problem = false
end

function update_state!(s::AdaptiveNewtonSolver, _, r)
    s.numiter += 1
    s.residuals[s.numiter] = r
    if s.numiter > 1
        s.reset_problem, s.update_type_nr = switch_information(s.switch_criterion, s)
    end
end

function maybe_reset_problem!(problem, Δa, s::AdaptiveNewtonSolver)
    if s.reset_problem
        map!(-, Δa, Δa)
        update_problem!(problem, Δa, get_update_spec(s))
    end
end

"""
    switch_information(switch_criterion, nlsolver)

Create a custom `switch_criterion` by overloading this function,
which given the defined `switch_criterion` and the `nlsolver`,
should return `reset_problem::Bool` and `new_nr::Int`, which 
determines if the problem should be reset to the state before its 
last update and which update_type should be used next, respectively.
"""
function switch_information end

# Implemented switches
"""
    NumIterSwitch(;switch_after)

Switch to `update_types[2]` after `switch_after`
iterations with `update_types[1]`
"""
@kwdef struct NumIterSwitch
    switch_after::Int
end

function switch_information(crit::NumIterSwitch, nls)
    numiter = getnumiter(nls)
    reset_problem = false
    new_nr = numiter > crit.switch_after ? 2 : 1
    return reset_problem, new_nr
end

"""
    NumIterSwitch(;switch_after)

Use `update_types[1]` when the convergence measure is larger than
`switch_at`. When the convergence measure is below `switch_at`, 
use `update_types[2]`. If the convergence measure increases above 
`switch_at` again, reset the problem and change to `update_types[1]`
iterations with `update_types[1]`.
"""
struct ToleranceSwitch{T}
    switch_at::T
end
function switch_information(crit::ToleranceSwitch, nls)
    k = getnumiter(nls)
    r = get_convergence_measures(nls, k)
    if r > crit.switch_at
        reset_problem = (get_type_number(nls) == 2)
        return reset_problem, 1
    else
        return false, 2
    end
end

"""
    IncreaseSwitch(;num_slow)

Use the (typically) fast but less stable `update_types[1]` as long 
as the convergence measure is decreasing. If it starts increasing, 
switch to the (typically) slower but more stable `update_types[2]`.
Use this for `num_slow` iterations after the convergence measure 
starts to decrease again.
"""
mutable struct IncreaseSwitch
    num_slow::Int # How many slow types to do after an increase in error
    num_at_switch::Int # At what numiter the switch was done
end
IncreaseSwitch(;num_slow) = IncreaseSwitch(num_slow, -(num_slow+1))
function switch_information(crit::IncreaseSwitch, nls)
    k = getnumiter(nls)
    Δr = get_convergence_measures(nls, k) - get_convergence_measures(nls, k-1)
    if Δr < 0 && (crit.num_at_switch+crit.num_slow < k) 
        return false, 1 # Use primary technique (typically faster, but less stable)
    else 
        Δr > 0 && (crit.num_at_switch = k) # Set num_at_switch if residual increasing
        reset_problem = (get_type_number(nls) == 1)
        return reset_problem, 2 # Use secondary technique (typically slower, but more stable)
    end
end