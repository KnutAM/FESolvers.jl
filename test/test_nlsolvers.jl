@testset "nlsolvers" begin
    tol = 1.e-6
    ls = ArmijoGoldstein(μ=1e-4, β=0.5, τ0=1.0)
    steepestdescent= SteepestDescent(;maxiter=30000, tolerance=1e-3, linesearch=ls)
    newton_ls = NewtonSolver(;maxiter=30, tolerance=tol, linesearch=ls)
    newton = NewtonSolver(;maxiter=30, tolerance=tol)
    custom = TestNLSolver(;maxiter=30, tolerance=tol, ls=ls)
    for nlsolver in [steepestdescent, newton_ls, newton, custom]
        problem = Rosenbrock() 
        converged = FerriteSolvers.solve_nonlinear!(nlsolver, problem)
        @test converged
        @test isapprox(norm(problem.r), 0.0; atol=1e-2)
        @test all(isapprox.(problem.x, 1.0; atol=1e-5))
    end

end