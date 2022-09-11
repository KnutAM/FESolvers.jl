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

""" getjacobian(problem)

Return the jacobian `drdx`, or approximations thereof
"""
function getjacobian end

""" getdescentpreconditioner(problem)

Return a preconditioner `K` for calculating the descent direction `p`,
considering solving `r(x)=0` as a minimization problem of `f(x)`
where `r=∇f`. The descent direction is then `p = K⁻¹ ∇f` 
"""
getdescentpreconditioner(::Any) = LinearAlgebra.I

"""
    getsystemmatrix(problem,solver)

Return the system matrix of the problem. For a Newton solver this
method should return the Jacobian, while for a steepest descent method
this can be a preconditioner as e.g., the L2 product of the gradients.
By default the system matrix for the `SteepestDescent` method is the unity matrix
and thus, renders a vanilla gradient descent solver.
"""
function getsystemmatrix end

"""
    calculate_energy(problem,𝐮)

Return the energy of the system (a scalar) which is the integrated energy
density over the domain Ω.
"""
function calculate_energy end

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
    calculate_convergence_measure(problem)

Calculate a value to be compared with the tolerance of the nonlinear solver. 
A standard case when using [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl)
is `norm(getresidual(problem)[Ferrite.free_dofs(dbcs)])` 
where `dbcs::Ferrite.ConstraintHandler`

"""
function calculate_convergence_measure end

"""
    check_convergence_criteria(problem, nlsolver)

Check if `problem` has converged and update the state 
of `nlsolver` wrt. number of iterations and a convergence
measure if applicable.
"""
function check_convergence_criteria(problem, nlsolver)
    r = calculate_convergence_measure(problem)
    update_state!(nlsolver, r)
    return r < gettolerance(nlsolver)
end

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
