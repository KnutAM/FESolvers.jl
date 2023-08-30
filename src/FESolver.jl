abstract type FESolver end


# Time-stepper related functions
function get_timestepper end

get_time(s::FESolver) = get_time(get_timestepper(s))
get_step(s::FESolver) = get_step(get_timestepper(s))

# Nonlinear solver related functions
function get_nlsolver end


# Functions related to both
function is_finished(s::FESolver)
    converged = is_converged(get_nlsolver(s))
    last_step = is_last_step(get_timestepper(s))
    return converged && last_step
end