mutable struct DummyNLSolver
    converged::Bool
end
FESolvers.is_converged(ds::DummyNLSolver) = ds.converged
function FESolvers.check_convergence(ds::DummyNLSolver)
    ds.converged && return nothing
    throw(FESolvers.ConvergenceError("DummyNLSolver not converged"))
end
FESolvers.get_num_iter(::DummyNLSolver) = 1
FESolvers.get_max_iter(::DummyNLSolver) = 4

function run_timestepper(solver, convergence_function=Returns(true))
    timestepper = FESolvers.get_timestepper(solver)
    FESolvers.reset_timestepper!(timestepper)
    t_old = FESolvers.get_time(timestepper)
    converged = true
    timehistory = [t_old]
    while !FESolvers.is_last_step(timestepper)
        FESolvers.step_time!(timestepper, DummyNLSolver(converged))
        t = FESolvers.get_time(timestepper)
        Δt = t - t_old
        converged = convergence_function(t, Δt, step)
        if converged
            t_old = t
            push!(timehistory, t)
        end
    end
    return timehistory
end



@testset "timesteppers" begin
    
    @testset "FixedTimeStepper" begin
        t = sort(rand(100))
        solver = QuasiStaticSolver(nothing, FixedTimeStepper(t))
        th = run_timestepper(solver, (args...) -> true)
        @test t ≈ th
        num_steps=10; Δt=0.1
        solver = QuasiStaticSolver(nothing, FixedTimeStepper(num_steps=num_steps, Δt=Δt))
        th = run_timestepper(solver, (args...) -> true)
        @test all( Δt .≈ (th[2:end]-th[1:end-1]))
        @test length(th) == (num_steps+1)
        t_avg = sum(t)/length(t)
        @test_throws FESolvers.ConvergenceError run_timestepper(solver, (time, args...) -> time < t_avg)
    end

    @testset "AdaptiveTimeStepper" begin
        # Errors 
        @test_throws ArgumentError AdaptiveTimeStepper(10.0, 0.1) # Flip Δt_init and t_end
        @test_throws ArgumentError AdaptiveTimeStepper(0.1, 10.0; t_start=20.0)
        @test_throws ArgumentError AdaptiveTimeStepper(0.1, 10.0; Δt_max=0.01)
        # Warnings
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; change_factor=1.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; change_factor=-0.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; optiter_ratio=1.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; optiter_ratio=-0.2)
        @test_logs (:warn,) match_mode=:any AdaptiveTimeStepper(0.1, 10.0; k=-0.2)

        t_end = 2.0
        Δt = 0.1
        Δt_min = 0.05
        Δt_max = 0.2

        ts = AdaptiveTimeStepper(Δt, t_end; Δt_min=Δt_min, Δt_max=Δt_max)
        solver = QuasiStaticSolver(DummyNLSolver(true), ts)
        @test FESolvers.get_time(ts) ≈ 0.0
        FESolvers.step_time!(solver)
        @test FESolvers.get_time(solver) ≈ Δt 
        @test FESolvers.get_step(solver) == 2

        cf(t, Δt, _) = t < 1.0 || Δt < (Δt_min+sqrt(eps(Δt_min)))

        th = run_timestepper(QuasiStaticSolver(DummyNLSolver(true),ts), cf)
        
        Δth = th[2:end]-th[1:end-1]
        @test Δth[1] .≈ Δt
        @test maximum(Δth) ≈ Δt_max 
        @test minimum(Δth[1:end-2]) .≈ Δt_min
        @test minimum(Δth) > Δt_min/2

        t_end = 0.91
        Δt = 0.1
        ts = AdaptiveTimeStepper(Δt, t_end; Δt_max = Δt)
        th = run_timestepper(QuasiStaticSolver(DummyNLSolver(true),ts), (args...)->true)
        Δth = th[2:end]-th[1:end-1]
        # Last two steps should take half each when the reminder is less then Δt_min
        @test Δth[end] ≈ Δth[end-1] ≈ (mod(t_end, Δt) + Δt)/2
        @test all(Δth[1:end-2] .≈ 0.1)

        # Test errors
        Δt = 0.1
        ts = AdaptiveTimeStepper(Δt, 3*Δt; Δt_min=Δt)
        # Not converged in first iteration (setup issue)
        @test_throws AssertionError FESolvers.step_time!(ts, DummyNLSolver(false))
        
        FESolvers.step_time!(ts, DummyNLSolver(true)) # Make one step
        # Cannot reduce time step further
        @test_throws FESolvers.ConvergenceError FESolvers.step_time!(ts, DummyNLSolver(false))
    end

end