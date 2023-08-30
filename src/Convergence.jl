# Convergence status
struct ConvergenceStatus
    converged::Bool
    reason::Symbol
    exception::Union{Exception,Nothing}
end
function ConvergenceStatus(converged::Bool)
    @assert converged "If not converged, reason should be given"
    return ConvergenceStatus(converged, :converged, nothing)
end
ConvergenceStatus(converged::Bool, reason) = ConvergenceStatus(converged, reason, nothing)

is_converged(cs::ConvergenceStatus) = cs.converged

# Exceptions
struct ConvergenceError <: Exception
    msg::String
end

function check_convergence(cs::ConvergenceStatus)
    is_converged(cs) && return nothing
    if isa(cs.exception, Exception)
        println(stderr, "FESolvers did not converge")
        printstyled(stderr, "Caught exception: ", bold=true, color=:yellow)
        showerror(stderr, cs.exception)
    end
    if cs.reason == :nonlinearsolver
        throw(ConvergenceError("The nonlinear solver did not converge"))
    elseif cs.reason == :linearsolver
        throw(ConvergenceError("The linear solver failed"))
    else
        throw(ConvergenceError("Not converged with an unknown reason, $(cs.reason), in ConvergenceStatus"))
    end
end