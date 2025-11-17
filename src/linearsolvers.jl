"""
    solve_linear!(Δx, K, r, linearsolver)

Using the method specified by `linearsolver`, 
solve `K Δx = -r` for `Δx`
"""
function solve_linear! end

"""
    BackslashSolver()

The standard julia linear solver using `Δx .= -K\\r`
"""
struct BackslashSolver end

solve_linear!(Δx, K, r, ::BackslashSolver) = map!(-, Δx, K\r)

"""
    LinearSolveSolver([alg], K, r=zeros(eltype(K), size(K,1)))

Create a linear solver with an algorithm `alg` from [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl).
Please see `LinearSolve.jl`'s [documentation](https://docs.sciml.ai/LinearSolve/stable/) for different solver algorithms. 
If not given, `LinearSolve`'s default algorithm will be used. 

!!! note "Extension"
    Using this solver requires `using` or `import`ing `LinearSolve.jl`

"""
struct LinearSolveSolver{C}
    cache::C
    algorithm::Symbol
    function LinearSolveSolver(alg, K::AbstractMatrix, r::AbstractVector=zeros(eltype(K), size(K, 1)); kwargs...)
        if !applicable(build_linear_solve_cache, alg, K, r; kwargs...)
            @warn "LinearSolve.jl required to use LinearSolveSolver, see the docstring"
        end
        cache = build_linear_solve_cache(alg, K, r; kwargs...)
        algorithm = alg === nothing ? :default : nameof(typeof(alg))
        return new{typeof(cache)}(cache, algorithm)
    end
end

# Overloaded in LinearSolveExt
function build_linear_solve_cache end