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
function FixedTimeStepper(num_steps::Int, Δt=1, t_start=zero(Δt))
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


"""
    AdaptiveTimeStepper(
        Δt_init::T, t_end::T; 
        t_start=zero(T), Δt_min=Δt_init, Δt_max=typemax(T), 
        change_factor=T(0.5), change_exponent=T(0.25)) where T

An adaptive time stepper with an initial step `Δt_init` and total  
time `t_end`. If the previous attempt did not converge, the time 
step is reduced as `Δt*=change_factor` and the step is retried. 
Otherwise, the next time step depends on how many iterations was 
required to converge, `numiter`. The time step is changed as 
`Δt*=change_factor^m`, where `m=(numiter-optiter)/(maxiter-optiter)`.

In this expression, `maxiter` and `optiter` are the maximum and optimum 
number of iterations for the nonlinear solver. 
These parameters are obtained from the functions `getnumiter(s)`,
`getmaxiter(s)`, and `getoptiter(s)` where `s` is the nonlinear solver. 
If not defined for the specific solver, `optiter=floor(maxiter/2)`.
"""
struct AdaptiveTimeStepper{T}
    t_start::T
    t_end::T
    Δt_init::T
    Δt_min::T
    Δt_max::T
    change_factor::T 
    Δt::ScalarWrapper{T}
end

function AdaptiveTimeStepper(
    Δt_init::T, t_end::T; 
    t_start=zero(T), Δt_min=Δt_init, Δt_max=typemax(T), 
    change_factor=T(0.5)) where T
    return AdaptiveTimeStepper(
        t_start, t_end, Δt_init, Δt_min, Δt_max,
        change_factor, ScalarWrapper(Δt_init))
end

initial_time(ts::AdaptiveTimeStepper) = ts.t_start 
islaststep(ts::AdaptiveTimeStepper, t, step) = t >= ts.t_end - eps(t)
function update_time(s::FerriteSolver{<:Any, <:AdaptiveTimeStepper}, t, step, converged)
    ts=s.timestepper
    
    # Initialization
    if step == 1 
        if !converged
            msg = "step=1 implies initial step and then \"convergence of the previous step\" must be true"
            throw(ArgumentError(msg))
        end
        ts.Δt[] = min(t+ts.Δt_init, ts.t_end)-t
        return t+ts.Δt[], step+1
    end

    if !converged
        if ts.Δt[] ≈ ts.Δt_min
            msg = "The nonlinear solve failed and the AdaptiveTimeStepper is at its minimum time step"
            throw(ConvergenceError(msg))
        end
        t -= ts.Δt[]
        ts.Δt[] = max(ts.Δt[]*ts.change_factor, ts.Δt_min)
    else
        numiter = getnumiter(s.nlsolver)
        optiter = getoptiter(s.nlsolver)
        maxiter = getmaxiter(s.nlsolver)
        m = (numiter-optiter)/(maxiter-optiter)
        ts.Δt[] = min(max(ts.Δt[] * (ts.change_factor^m), ts.Δt_min), ts.Δt_max)
        step += 1
    end

    # Ensure that the last time step is not too short.
    # With the following algorithm, the last two time steps
    # are only guaranteed to be > Δt_min/2
    t_remaining = ts.t_end - (t+ts.Δt[])
    if t_remaining < eps(t)
        ts.Δt[] = ts.t_end-t
        t = ts.t_end
    elseif t_remaining < ts.Δt_min
        ts.Δt[] = (ts.t_end - t)/2
        t += ts.Δt[]
    else
        t += ts.Δt[]
    end

    return t, step
end
