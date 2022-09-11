"""
    update_guess!(Δx, drdx, r, linearsolver)

Using the method specified by `linearsolver`, 
solve `r + drdx * Δx = 0` for `Δx`
"""
function solve_linear! end

"""
    BackslashSolver()

The standard julia linear solver using `Δx = -drdx\r`
"""
struct BackslashSolver end

solve_linear!(Δa, K, r, ::BackslashSolver) = (Δa .= -K\r)

