"""
    get_time(timestepper)

Return the current time for `timestepper`
"""
function get_time end

"""
    get_step(timestepper)

Return the current step number
"""
function get_step end

"""
    is_last_step(timestepper)->Bool

Return `true` if the current `step`/`time` is the last step,
return `false` otherwise 
"""
function is_last_step end

"""
    step_time!(solver)
    step_time!(timestepper, nlsolver)

Increment the `timestepper` depending on the convergence status of `nlsolver`.
If not converged and a smaller time step is not possible, throw `ConvergenceError`. 

Note that a call to the first definition is forwarded to the second function definition 
by decomposing the solver, unless another specialization is defined.
"""
function step_time! end

step_time!(s::FESolver) = step_time!(get_timestepper(s), get_nlsolver(s))

"""
    reset_timestepper!(timestepper)

Reset the time and step in `timestepper`
"""
function reset_timestepper! end

"""
    FixedTimeStepper(num_steps::int, Δt, t_start=0)
    FixedTimeStepper(t::Vector)

A time stepper which gives fixed time steps. If the convenience
interface is used, constant increments are used. Note that 
`length(t)=num_steps+1` since the first value is just the initial 
value and is not an actual step.  
"""
mutable struct FixedTimeStepper{T}
    const t::Vector{T}
    step::Int
end
FixedTimeStepper(t::Vector) = FixedTimeStepper(t, 1)
function FixedTimeStepper(;num_steps::Int, Δt=1, t_start=zero(Δt))
    return FixedTimeStepper(t_start .+ collect(0:Δt:((num_steps)*Δt)))
end

get_time(ts::FixedTimeStepper) = ts.t[ts.step]
get_step(ts::FixedTimeStepper) = ts.step
is_last_step(ts::FixedTimeStepper) = ts.step >= length(ts.t)
function step_time!(ts::FixedTimeStepper, nlsolver)
    check_convergence(nlsolver)
    ts.step += 1
end
function reset_timestepper!(ts::FixedTimeStepper)
    ts.step = 1
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
the nonlinear solver (via `get_max_iter(s)`)

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
    step::Int
    time::T
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
        change_factor, optiter_ratio, k, Δt_init, 1, t_start)
end

get_time(ts::AdaptiveTimeStepper) = ts.time
get_step(ts::AdaptiveTimeStepper) = ts.step
is_last_step(ts::AdaptiveTimeStepper) = ts.time >= ts.t_end - eps(ts.t_end)
function reset_timestepper!(ts::AdaptiveTimeStepper)
    ts.Δt = ts.Δt_init
    ts.step = 1
    ts.time = ts.t_start
end

function step_time!(ts::AdaptiveTimeStepper, nlsolver)
    # Initialization
    if ts.step == 1
        @assert is_converged(nlsolver)
        ts.Δt = min(ts.time+ts.Δt_init, ts.t_end) - ts.time
        ts.time += ts.Δt
        ts.step += 1
        return nothing
    end

    (ts.Δt ≈ ts.Δt_min) && check_convergence(nlsolver)

    if is_converged(nlsolver)
        numiter = get_num_iter(nlsolver)
        maxiter = get_max_iter(nlsolver)
        optiter = Int(floor(ts.optiter_ratio*maxiter))
        m = (numiter-optiter)/(maxiter-optiter)
        ts.Δt = min(max(ts.Δt * (ts.change_factor^m), ts.Δt_min), ts.Δt_max)
        ts.step += 1
    else
        ts.time -= ts.Δt
        ts.Δt = max(ts.Δt*ts.change_factor, ts.Δt_min)
    end

    # Ensure that the last time step is not too short.
    # With the following algorithm, the last two time steps
    # are only guaranteed to be > Δt_min/2
    t_remaining = ts.t_end - (ts.time + ts.Δt)
    if t_remaining < eps(ts.time)
        ts.Δt = ts.t_end - ts.time
        ts.time = ts.t_end
    elseif t_remaining < ts.Δt_min
        ts.Δt = (ts.t_end - ts.time)/2
        ts.time += ts.Δt
    else
        ts.time += ts.Δt
    end

    return nothing
end
