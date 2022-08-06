"""
    getunknowns(problem)

Return the current vector of unknown values
"""
function getunknowns end

"""
    getresidual(problem)

Return the current residual vector of the problem
"""
function getresidual end

"""
    getjacobian(problem)

Return the current jacobian of the problem
"""
function getjacobian end

"""
    update_to_next_step!(problem, time)

Update prescribed values, external loads etc. for the given time
Called in the beginning of each new time step. 
Note: For adaptive time stepping, it may be called with a lower 
time than the previous time if the solution did not converge.
"""
function update_to_next_step! end

"""
    update_problem!(problem, Δx=0*getunknowns(problem))

Assemble the residual and stiffness for `x+=Δx`. 

- Some linear solvers may be inaccurate, and if modified stiffness is used 
  to enforce constraints on `x`, it is good the force `Δx=0` on these
  components inside this function. 
- Note that the function must also support only one argument: `problem`,
  this version is called the first time after 
  `update-update_to_next_step!` and should default to `Δx=0`
"""
function update_problem! end


"""
    calculate_residualnorm(problem)

Calculate the value of the residual norm to be compared with the iteration tolerance
"""
function calculate_residualnorm end

"""
    handle_converged!(problem)

Do necessary update operations once it is known that the 
problem has converged. E.g., update old values to the current. 
Only called directly after the problem has converged. 
"""
function handle_converged! end

"""
    postprocess!(problem, step)

Perform any postprocessing at the current time and step nr `step`
Called after time step converged, and after `handle_converged!`
"""
function postprocess! end
