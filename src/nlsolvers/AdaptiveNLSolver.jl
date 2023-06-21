abstract type AbstractAdaptiveNLSolver end

# Functions required by an AbstractAdaptiveNLSolver wrapper
function reset_solver! end
function is_finished end
function is_converged end
function update_solver end
function get_all_nlsolvers end

# Main solver function
function solve_nonlinear!(problem, nlsolvers::AbstractAdaptiveNLSolver)
    x0 = copy(getunknowns(problem))
    reset_solver!(nlsolvers)
    while !is_finished(nlsolvers)
        nlsolver = update_solver(nlsolvers)
        should_reset_problem(nlsolvers)     && reset_problem!(problem, nlsolver; x=x0)
        solve_nonlinear!(problem, nlsolver)
    end
    return nothing
end

# Additional functions to make it work like a regular nlsolver from the user point of view
get_max_iter(s::AbstractAdaptiveNLSolver) = sum(get_max_iter, get_all_nlsolvers(s))
get_num_iter(s::AbstractAdaptiveNLSolver) = sum(get_num_iter, get_all_nlsolvers(s))
get_initial_update_spec(s::AbstractAdaptiveNLSolver) = get_initial_update_spec(first(get_all_nlsolvers(s)))

### ======================================================================= ###
### MultiStageSolver
### ======================================================================= ###
"""
    MultiStageSolver

Solve a problem with multiple stages of nonlinear solvers, switching when each solver 
finishes.
"""
mutable struct MultiStageSolver{ST} <: AbstractAdaptiveNLSolver
    const nlsolvers::Vector{ST}
    const require_converged::Vector{Bool}
    solver_id::Int
end
function MultiStageSolver(nlsolvers, require_converged=true)
    make_vector(_, tovec::Vector) = tovec
    make_vector(s, tovec::Bool) = [tovec for _ in 1:(length(s)-1)]
    return MultiStageSolver(nlsolvers, make_vector(nlsolvers, require_converged), 0)
end
get_current_nlsolver(s::MultiStageSolver) = s.nlsolvers[s.solver_id]
get_all_nlsolvers(s::MultiStageSolver) = s.nlsolvers

function reset_solver!(s::MultiStageSolver)
    s.solver_id = 0
end
function is_finished(s::MultiStageSolver)
    s.solver_id == 0                        && return false # Hasn't started
    s.solver_id == length(s.nlsolvers)      && return true # Last nlsolver
    is_converged(get_current_nlsolver(s))   && return false # Converged, but not the last
    s.require_converged[s.solver_id]        && return true # Not converged, but that is required
end

function is_converged(s::MultiStageSolver)
    return is_finished(s) && is_converged(get_current_nlsolver(s))
end
function update_solver(s::MultiStageSolver)
    s.solver_id += 1
    return get_current_nlsolver(s)
end

### ======================================================================= ###
### DynamicSolver
### ======================================================================= ###
"""
    DynamicSolver(nlsolver, updater)

`DynamicSolver` contains a base `nlsolver` that must support `set_update_type!` such that 
the value of `type` in `UpdateSpec` can be changed dynamically. This is typically used to change
regularization factors for the jacobian calculation. `updater` should be a function with the 
signature `type, reset, finished = updater(nlsolver, num_attempts)`, 
where `type` is given to `UpdateSpec` in `nlsolver`, `reset` says if the problem should be reset,
and `finished` tells if this is the last update that can be done (and if not converged then, it will fail).
The inputs are the current `nlsolver` as well as the number of attempts. 
"""
mutable struct DynamicSolver{NLS,U} <: AbstractAdaptiveNLSolver
    const nlsolver::NLS
    const updater::U # Function type, reset, finished = updater(nlsolver, num_attempts)
    should_reset::Bool
    num_attempts::Int
    is_finished::Bool
end
DynamicSolver(nlsolver, updater) = DynamicSolver(nlsolver, updater, false, 0, false)
function reset_solver!(s::DynamicSolver)
    s.should_reset = false
    s.num_attempts = 0
    s.is_finished = false
end

is_finished(s::DynamicSolver) = is_converged(s) || s.is_finished
is_converged(s::DynamicSolver) = is_converged(s.nlsolver)
function update_solver(s::DynamicSolver)
    s.num_attempts += 1
    type, s.should_reset, s.is_finished = s.updater(s.nlsolver, s.num_attempts)
    set_update_type!(s.nlsolver, type)
    return s.nlsolver
end
#get_current_nlsolver(s::DynamicSolver) = s.nlsolver
get_all_nlsolvers(s::DynamicSolver) = (s.nlsolver,)
