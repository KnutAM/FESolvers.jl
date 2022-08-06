"""
    update_guess!(Δx, drdx, r, linearsolver)

Using the method specified by `linearsolver`, 
solve `r + drdx * Δx = 0` for `Δx`
"""
function update_guess! end

"""
    BackslashSolver()

The standard julia linear solver using `Δx = -drdx\r`
"""
struct BackslashSolver end

update_guess!(Δa, K, r, ::BackslashSolver) = (Δa .= -K\r)

