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

solve_linear!(Δx, K, r, ::BackslashSolver) = (Δx .= -K\r)

