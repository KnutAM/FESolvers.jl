# Time steppers
The following time steppers are implemented in FESolvers.

```@docs
FixedTimeStepper
AdaptiveTimeStepper
```

## Custom time stepper
A time stepper should support the following functions

```@docs
FESolvers.get_time
FESolvers.get_step
FESolvers.is_last_step
FESolvers.step_time!
FESolvers.reset_timestepper!
```