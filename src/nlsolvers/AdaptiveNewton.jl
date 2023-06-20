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
Base.@kwdef mutable struct AdaptiveNewtonSolver{LinearSolver,LineSearch,T,UT,SC,SS}
    const linsolver::LinearSolver = BackslashSolver()
    const linesearch::LineSearch = NoLineSearch()
    const maxiter::Int = 10
    const tolerance::T = 1e-6
    const update_jac_first::Bool=true # Should jacobian be updated first iteration (or use the one from the previous converged)
    const update_types::UT
    update_type_nr::Int=1
    const switch_criterion::SC
    reset_problem::Bool=false
    const state::SS=SolverState(maxiter)
end
getsystemmatrix(problem, ::AdaptiveNewtonSolver) = getjacobian(problem)

_get_update_type(nls::AdaptiveNewtonSolver) = nls.update_types[nls.update_type_nr]
function get_initial_update_spec(nls::AdaptiveNewtonSolver)
    return UpdateSpec(;jacobian=!(nls.update_jac_first), residual=false, type=_get_update_type(nls))
end
function get_first_update_spec(nls::AdaptiveNewtonSolver)
    if is_converged(nls)
        return UpdateSpec(jacobian=nls.update_jac_first, residual=true, type=_get_update_type(nls))
    else # We must update since jacobian when it didn't converge
        return UpdateSpec(jacobian=true, residual=true, type=_get_update_type(nls))
    end
end
function get_update_spec(nls::AdaptiveNewtonSolver)
    return UpdateSpec(;jacobian=true, residual=true, type=_get_update_type(nls))
end

get_max_iter(nlsolver::AdaptiveNewtonSolver) = nlsolver.maxiter
get_tolerance(nlsolver::AdaptiveNewtonSolver) = nlsolver.tolerance

get_linesearch(nlsolver::AdaptiveNewtonSolver) = nlsolver.linesearch
get_linear_solver(nlsolver::AdaptiveNewtonSolver) = nlsolver.linsolver
get_solver_state(nlsolver::AdaptiveNewtonSolver) = nlsolver.state

get_type_number(s::AdaptiveNewtonSolver) = s.update_type_nr

function reset_solver_state!(s::AdaptiveNewtonSolver)
    reset_solver_state!(get_solver_state(s))
    s.update_type_nr = 1
    s.reset_problem = false
end

function update_solver_state!(s::AdaptiveNewtonSolver, _, r)
    update_solver_state!(get_solver_state(s), s, r)
    if get_num_iter(s) > 1
        s.reset_problem, s.update_type_nr = switch_information(s.switch_criterion, s)
    end
end

should_reset_problem(s::AdaptiveNewtonSolver) = s.reset_problem

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
    numiter = get_num_iter(nls)
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
    k = get_num_iter(nls)
    r = get_convergence_measure(nls, k)
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
    k = get_num_iter(nls)
    Δr = get_convergence_measure(nls, k) - get_convergence_measure(nls, k-1)
    if Δr < 0 && (crit.num_at_switch+crit.num_slow < k) 
        return false, 1 # Use primary technique (typically faster, but less stable)
    else 
        Δr > 0 && (crit.num_at_switch = k) # Set num_at_switch if residual increasing
        reset_problem = (get_type_number(nls) == 1)
        return reset_problem, 2 # Use secondary technique (typically slower, but more stable)
    end
end