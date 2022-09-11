"""
    initial_time(timestepper)

Return the starting time for the given `timestepper`
"""
function initial_time end

"""
    islaststep(timestepper, time, step)->Bool

Return `true` if the current `step`/`time` is the last step,
return `false` otherwise 
"""
function islaststep end

"""
    update_time(solver::FerriteSolver{<:Any, <:TST}, time, step, converged::Bool)
Return the next time and step number, depending on if the previous time step converged 
or not. If not converged, return the same `step` but a `new_time<time` to reduce the 
time step. If it is not possible to retry with shorter timestep, throw 
`ConvergenceError`. If converged, update time step as planned. 
Note: The full solver is given as input to allow specialization on e.g. if a 
Newton iteration has required many iterations, shorten the next time step as a 
precausionary step
"""
function update_time end

"""
    FixedTimeStepper(num_steps::int, Δt, t_start=0)
    FixedTimeStepper(t::Vector)

A time stepper which gives fixed time steps. If the convenience
interface is used, constant increments are used. Note that 
`length(t)=num_steps+1` since the first value is just the initial 
value and is not an actual step.  
"""
struct FixedTimeStepper{T}
    t::Vector{T}
end
function FixedTimeStepper(;num_steps::Int, Δt=1, t_start=zero(Δt))
    return FixedTimeStepper(t_start .+ collect(0:Δt:((num_steps)*Δt)))
end

initial_time(ts::FixedTimeStepper) = first(ts.t)
islaststep(ts::FixedTimeStepper, _, step) = step >= length(ts.t)
function update_time(s::FerriteSolver{<:Any,<:FixedTimeStepper}, t, step, converged)
    if !converged
        msg = "nonlinear solve failed and FixedTimeStepper cannot adjust time step"
        throw(ConvergenceError(msg))
    end
    return s.timestepper.t[step+1], step+1
end