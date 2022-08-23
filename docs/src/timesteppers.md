# Time steppers
The following time steppers are included

```@docs
FixedTimeStepper
AdaptiveTimeStepper
```

## Implementation of custom time steppers
A time stepper should support the following functions

```@docs
FerriteSolvers.initial_time
FerriteSolvers.islaststep
FerriteSolvers.update_time
```
