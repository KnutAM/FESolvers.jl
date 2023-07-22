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
    update_time(solver, time, step, converged::Bool)
    update_time(timestepper, nlsolver, time, step, converged::Bool)

Return the next time and step number, depending on if the previous time step converged 
or not. If not converged, return the same `step` but a `new_time<time` to reduce the 
time step. If it is not possible to retry with shorter timestep, throw 
`ConvergenceError`. If converged, update time step as planned. 
Note: The full solver is given as input to allow specialization on e.g. if a 
Newton iteration has required many iterations, shorten the next time step as a 
precausionary step.

Note that a call to the first definition is forwarded to the second function definition 
by decomposing the solver, unless another specialization is defined.
"""
function update_time end

update_time(s::FESolver, args...; kwargs...) = update_time(gettimestepper(s), getnlsolver(s), args...; kwargs...)

"""
    FixedTimeStepper(;num_steps::Int, Δt=1, t_start=0)
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
function update_time(ts::FixedTimeStepper, nlsolver, t, step, converged)
    if !converged
        msg = "nonlinear solve failed and FixedTimeStepper cannot adjust time step"
        throw(ConvergenceError(msg))
    end
    return ts.t[step+1], step+1
end


"""
    AdaptiveTimeStepper(
        Δt_init::T, t_end::T; 
        t_start=zero(T), Δt_min=Δt_init, Δt_max=typemax(T), 
        change_factor=T(0.5), optiter_ratio=T(0.5), k=one(T)) where T

An adaptive time stepper with an initial step `Δt_init` and total  
time `t_end`. Two ways of adaption:

1. If the previous attempt did not converge, the time 
step is reduced as `Δt*=change_factor` and the step is retried. 

2. If convergence, the next time step depends on how many iterations was 
required to converge; `numiter`. The time step is changed as 
`Δt*=change_factor^(k*m)`, where `m=(numiter-optiter)/(maxiter-optiter)`.
In this expression, `maxiter` and `optiter` are the maximum and optimum 
number of iterations for the nonlinear solver. 
`optiter=floor(maxiter*optiter_ratio)` and `maxiter` is obtained from 
the nonlinear solver (via `getmaxiter(s)`)

If `numiter=maxiter`, then `m=1` and the time step update is the same as
for a non-converged solution if `k=1`. Note that `k>0`, `change_factor∈[0,1]`,
and `optiter_ratio∈[0,1]` are expected, otherwise warnings are thrown. 
"""
mutable struct AdaptiveTimeStepper{T}
    const t_start::T
    const t_end::T
    const Δt_init::T
    const Δt_min::T
    const Δt_max::T
    const change_factor::T
    const optiter_ratio::T
    const k::T
    Δt::T
end

function AdaptiveTimeStepper(
    Δt_init::T, t_end::T; 
    t_start=zero(T), Δt_min=Δt_init, Δt_max=typemax(T), 
    change_factor::T=T(0.5), optiter_ratio::T=T(0.5), k=one(T)) where T
    # Checks
    t_start > t_end && throw(ArgumentError("t_start=$t_start must be < t_end=$t_end"))
    Δt_min > Δt_max && throw(ArgumentError("Δt_min=$Δt_min must be < Δt_max=$Δt_max"))
    if Δt_min > (t_end-t_start)
        throw(ArgumentError("Δt_min=$Δt_min must be >= t_end-t_start=$(t_end-t_start)"))
    end
    
    if !(0<change_factor<1)
        @warn "change_factor=$change_factor ∉ [0,1] ⇒ strange adaptivity behavior expected"
    end
    if !(0<optiter_ratio<1)
        @warn "optiter_ratio=$optiter_ratio ∉ [0,1] ⇒ strange adaptivity behavior expected"
    end
    k<0 && @warn "k=$k < 0 ⇒ strange adaptivity behavior expected"

    return AdaptiveTimeStepper(
        t_start, t_end, Δt_init, Δt_min, Δt_max,
        change_factor, optiter_ratio, k, Δt_init)
end

initial_time(ts::AdaptiveTimeStepper) = ts.t_start 
islaststep(ts::AdaptiveTimeStepper, t, step) = t >= ts.t_end - eps(t)
function update_time(ts::AdaptiveTimeStepper, nlsolver, t, step, converged)
    # Initialization
    if step == 1 
        if !converged
            msg = "step=1 implies initial step and then \"convergence of the previous step\" must be true"
            throw(ArgumentError(msg))
        end
        ts.Δt = min(t+ts.Δt_init, ts.t_end)-t
        return t+ts.Δt, step+1
    end

    if !converged
        if ts.Δt ≈ ts.Δt_min
            msg = "The nonlinear solve failed and the AdaptiveTimeStepper is at its minimum time step"
            throw(ConvergenceError(msg))
        end
        t -= ts.Δt
        ts.Δt = max(ts.Δt*ts.change_factor, ts.Δt_min)
    else
        numiter = getnumiter(nlsolver)
        maxiter = getmaxiter(nlsolver)
        optiter = Int(floor(ts.optiter_ratio*maxiter))
        m = (numiter-optiter)/(maxiter-optiter)
        ts.Δt = min(max(ts.Δt * (ts.change_factor^m), ts.Δt_min), ts.Δt_max)
        step += 1
    end

    # Ensure that the last time step is not too short.
    # With the following algorithm, the last two time steps
    # are only guaranteed to be > Δt_min/2
    t_remaining = ts.t_end - (t+ts.Δt)
    if t_remaining < eps(t)
        ts.Δt = ts.t_end-t
        t = ts.t_end
    elseif t_remaining < ts.Δt_min
        ts.Δt = (ts.t_end - t)/2
        t += ts.Δt
    else
        t += ts.Δt
    end

    return t, step
end