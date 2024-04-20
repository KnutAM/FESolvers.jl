module LinearSolveExt
import FESolvers
import LinearSolve

FESolvers.LinearSolveSolver(K::AbstractMatrix, args...) = FESolvers.LinearSolveSolver(nothing, K, args...)

function FESolvers.build_linear_solve_cache(alg::Union{LinearSolve.SciMLLinearSolveAlgorithm, Nothing}, K, r)
    return LinearSolve.init(LinearSolve.LinearProblem(K, r), alg)
end

function FESolvers.solve_linear!(Δx, K, r, solver::FESolvers.LinearSolveSolver)
    cache = solver.cache
    # Update matrices in cache
    cache.A = K
    cache.b = r 
    # Solve problem
    solution = LinearSolve.solve!(cache)
    # Update Δx
    map!(-, Δx, solution.u)
    return Δx
end

function Base.show(io::IO, ::MIME"text/plain", lss::FESolvers.LinearSolveSolver)
    print(io, "LinearSolveSolver(:", lss.algorithm, ")")
end

end