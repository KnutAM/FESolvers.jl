# Nonlinear solvers
```@docs
NewtonSolver
FESolvers.AdaptiveNewtonSolver
SteepestDescent
LinearProblemSolver
```

## Further details on selected solvers
### AdaptiveNewtonSolver
For the adaptive newton solver, a few basic switchers have been implemented.
```@docs
FESolvers.NumIterSwitch
FESolvers.ToleranceSwitch
FESolvers.IncreaseSwitch
```

To implement a custom switcher for `AdaptiveNewtonSolver`, define a new struct
and the function `switch_information` for that struct:
```@docs
FESolvers.switch_information
```

## Problem update specification
An `UpdateSpec`, which can be querried for information,
is passed to [`update_problem!`](@ref FESolvers.update_problem!) to give instructions on 
how to update the problem. 
```@docs
FESolvers.UpdateSpec
```

## Custom solvers
A custom nonlinear solver can be written by tapping into the existing functions 
at different levels. For example, `LinearProblemSolver` defines a custom 
[`solve_nonlinear!`](@ref FESolvers.solve_nonlinear!) that uses the 
default [`calculate_update!`](@ref FESolvers.calculate_update!).

All nonlinear solvers are expected to implement the following methods,
```@docs
FESolvers.getmaxiter
FESolvers.getnumiter
FESolvers.get_convergence_measures
FESolvers.gettolerance
FESolvers.get_initial_update_spec
```

After implementing these, one can either implement `solve_nonlinear!`,
or a set of method described below to use the default implementation.
```@docs
FESolvers.solve_nonlinear!
```

### Methods required by `solve_nonlinear!`
To use the default implementation of `solve_nonlinear!`,
[`calculate_update!`](@ref FESolvers.calculate_update!) and the other methods in the
list below must be implemented for the specific nonlinear solver. 
The default implementation of `calculate_update!` may be used as well,
see the description below.
```@docs
FESolvers.calculate_update!
FESolvers.update_state!
FESolvers.reset_state!
FESolvers.get_first_update_spec
FESolvers.get_update_spec
FESolvers.maybe_reset_problem!
```

### Methods required by `calculate_update!`
To support the default `calculate_update!` implementation, 
the following methods must be implemented for the given solver. 
```@docs
FESolvers.get_linear_solver
FESolvers.get_linesearch
```

### Additional methods that usually don't require specialization
```@docs
FESolvers.do_initial_update
```

# Linesearch
Some nonlinear solvers can use linesearch as a complement, 
and the following linesearches are included. 
```@docs
NoLineSearch
ArmijoGoldstein
```

## Custom linesearch
A custom linesearch should implement the following function
```@docs
FESolvers.linesearch!
```