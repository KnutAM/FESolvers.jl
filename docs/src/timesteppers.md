# Time steppers
The following time steppers are implemented in FESolvers.

```@docs
FixedTimeStepper
AdaptiveTimeStepper
```

## Custom time stepper
A time stepper should support the following functions

```@docs
FESolvers.initial_time
FESolvers.islaststep
FESolvers.update_time
```