function run_timestepper(solver, convergence_function)
    t = FerriteSolvers.initial_time(solver.timestepper)
    t_old = t
    step = 1
    converged = true
    timehistory = [t]
    while !FerriteSolvers.islaststep(solver.timestepper, t, step)
        t, step = FerriteSolvers.update_time(solver, t, step, converged)
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
    #=
    @testset "FixedTimeStepper" begin
        t = sort(rand(100))
        solver = FerriteSolver(nothing, FixedTimeStepper(t))
        th = run_timestepper(solver, (args...) -> true)
        @test t ≈ th

        @test_throws FerriteSolvers.ConvergenceError run_timestepper(solver, (time, args...) -> time < sum(t)/length(t))
    end
    =#
    @testset "AdaptiveTimeStepper" begin
        t_end = 2.0
        Δt = 0.1
        Δt_min = 0.05
        Δt_max = 0.2
        nc2incr = 2
        ts = AdaptiveTimeStepper(Δt, t_end; Δt_min=Δt_min, Δt_max=Δt_max, num_converged_to_increase=nc2incr)
        solver = FerriteSolver(nothing, ts)
        @test FerriteSolvers.initial_time(ts) ≈ 0.0
        t, step = FerriteSolvers.update_time(solver, 0.0, 1, true)
        @test t ≈ Δt 
        @test step == 2

        cf(t, Δt, _) = t < 1.0 || Δt < (Δt_min+sqrt(eps(Δt_min)))
        th = run_timestepper(FerriteSolver(nothing,ts), cf)
        Δth = th[2:end]-th[1:end-1]
        @test all(Δth[1:2] .≈ Δt )
        @test maximum(Δth) ≈ Δt_max 
        @test all(minimum(Δth[1:end-2]) .≈ Δt_min)
        @test minimum(Δth) > Δt_min/2

        t_end = 0.91
        Δt = 0.1
        ts = AdaptiveTimeStepper(Δt, t_end; Δt_max = Δt)
        th = run_timestepper(FerriteSolver(nothing,ts), (args...)->true)
        Δth = th[2:end]-th[1:end-1]
        # Last two steps should take half each when the reminder is less then Δt_min
        @test Δth[end] ≈ Δth[end-1] ≈ (mod(t_end, Δt) + Δt)/2
        @test all(Δth[1:end-2] .≈ 0.1)
    end

end