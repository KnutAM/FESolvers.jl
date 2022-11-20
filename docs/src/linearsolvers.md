# Linear solvers
A linear solver should support the `solve_linear!` function specified below. 

```@docs
FESolvers.solve_linear!
```

## Implemented Solvers

### BackslashSolver
```@docs
BackslashSolver
```

### `LinearSolve.jl`
The linear solvers in [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl) are available 
if the `LinearSolve.jl` package is available (implemented via [`Requires.jl`](https://github.com/JuliaPackaging/Requires.jl)).
This also includes their default solver that is supplied setting the linear solver to `nothing`. 
Please see `LinearSolve.jl`'s [documentation](https://docs.sciml.ai/LinearSolve/stable/) for their different solvers. 