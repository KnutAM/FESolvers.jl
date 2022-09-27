var documenterSearchIndex = {"docs":
[{"location":"solvers/#Solvers","page":"Solvers","title":"Solvers","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Solvers are the top level object in this package,  each solver should define the function ","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"FESolvers.solve_problem!","category":"page"},{"location":"solvers/#Implemented-Solvers","page":"Solvers","title":"Implemented Solvers","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"QuasiStaticSolver","category":"page"},{"location":"nlsolvers/#Nonlinear-solvers","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"","category":"section"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"A nonlinear solver should support the solve_nonlinear! function specified below. ","category":"page"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"FESolvers.solve_nonlinear!","category":"page"},{"location":"nlsolvers/#FESolvers.solve_nonlinear!","page":"Nonlinear solvers","title":"FESolvers.solve_nonlinear!","text":"solve_nonlinear!(nlsolver, problem)\n\nSolve the current time step in the nonlinear problem, (r(x) = 0), by using the nonlinear solver: nlsolver. \n\n\n\n\n\n","category":"function"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"It can do so, by supporting the following functions","category":"page"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"FESolvers.calculate_update\nFESolvers.getmaxiter\nFESolvers.gettolerance","category":"page"},{"location":"nlsolvers/#FESolvers.calculate_update","page":"Nonlinear solvers","title":"FESolvers.calculate_update","text":"function calculate_update(problem, nlsolver, iter)\n\nAccording to the nonlinear solver, nlsolver, at iteration iter, calculate the update, Δx to the unknowns x.\n\n\n\n\n\n","category":"function"},{"location":"nlsolvers/#FESolvers.getmaxiter","page":"Nonlinear solvers","title":"FESolvers.getmaxiter","text":"getmaxiter(nlsolver)\n\nReturns the maximum number of iterations allowed for the nonlinear solver\n\n\n\n\n\n","category":"function"},{"location":"nlsolvers/#FESolvers.gettolerance","page":"Nonlinear solvers","title":"FESolvers.gettolerance","text":"gettolerance(nlsolver)\n\nReturns the iteration tolerance for the solver\n\n\n\n\n\n","category":"function"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"and optionally","category":"page"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"FESolvers.update_state!\nFESolvers.reset_state!","category":"page"},{"location":"nlsolvers/#FESolvers.update_state!","page":"Nonlinear solvers","title":"FESolvers.update_state!","text":"update_state!(nlsolver, r)\n\nA nonlinear solver may solve information about its convergence state. r is the output from calculate_convergence_measure when  this function is called by the default implementation of  check_convergence_criteria.  update_state! is optional to implement\n\n\n\n\n\n","category":"function"},{"location":"nlsolvers/#FESolvers.reset_state!","page":"Nonlinear solvers","title":"FESolvers.reset_state!","text":"reset_state!(nlsolver)\n\nIf update_state! is implemented, this function is used to  reset its state at the beginning of each new time step. \n\n\n\n\n\n","category":"function"},{"location":"nlsolvers/#Implemented-Solvers","page":"Nonlinear solvers","title":"Implemented Solvers","text":"","category":"section"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"NewtonSolver\nSteepestDescent","category":"page"},{"location":"nlsolvers/#FESolvers.NewtonSolver","page":"Nonlinear solvers","title":"FESolvers.NewtonSolver","text":"NewtonSolver(;linsolver=BackslashSolver(), linesearch=NoLineSearch(), maxiter=10, tolerance=1.e-6)\n\nUse the standard NewtonRaphson solver to solve the nonlinear  problem r(x) = 0 with tolerance within the maximum number  of iterations maxiter. The linsolver argument determines the used linear solver whereas the linesearch can be set currently between NoLineSearch or ArmijoGoldstein. The latter globalizes the Newton strategy.\n\n\n\n\n\n","category":"type"},{"location":"nlsolvers/#FESolvers.SteepestDescent","page":"Nonlinear solvers","title":"FESolvers.SteepestDescent","text":"SteepestDescent(;maxiter=10, tolerance=1.e-6)\n\nUse a steepest descent solver to solve the nonlinear  problem r(x) = 0, which minimizes a potential \\Pi with tolerance and the maximum number of iterations maxiter.\n\nThis method is second derivative free and is not as locally limited as a Newton-Raphson scheme. Thus, it is especially suited for stronlgy nonlinear behavior with potentially vanishing tangent stiffnesses.\n\n\n\n\n\n","category":"type"},{"location":"nlsolvers/#Linesearch","page":"Nonlinear solvers","title":"Linesearch","text":"","category":"section"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"Some nonlinear solvers can use linesearch as a complement,  and the following linesearches are included. ","category":"page"},{"location":"nlsolvers/","page":"Nonlinear solvers","title":"Nonlinear solvers","text":"NoLineSearch\nArmijoGoldstein","category":"page"},{"location":"nlsolvers/#FESolvers.NoLineSearch","page":"Nonlinear solvers","title":"FESolvers.NoLineSearch","text":"Singleton that does not perform a linesearch when used in a nonlinear solver\n\n\n\n\n\n","category":"type"},{"location":"nlsolvers/#FESolvers.ArmijoGoldstein","page":"Nonlinear solvers","title":"FESolvers.ArmijoGoldstein","text":"Armijo-Goldstein{T}(;β=0.9,μ=0.01,τ0=1.0,τmin=1e-4)\n\nBacktracking line search based on the Armijo-Goldstein condition\n\nPi(boldsymbolu + tau Deltaboldsymbolu) leq Pi(boldsymbolu) - mutaudeltaPi(boldsymbolu)Delta boldsymbolu\n\nwhere $\\Pi$ is the potential, $\\tau$ the stepsize, and $\\delta\\Pi$ the residuum.\n\n#Fields\n\nβ::T = 0.9 constant factor that changes the steplength τ in each iteration\nμ::T = 0.01 second constant factor that determines how much the potential needs to decrease additionally\nτ0::T = 1.0 start stepsize \nτmin::T = 1e-4 minimal stepsize\n\n\n\n\n\n","category":"type"},{"location":"linearsolvers/#Linear-solvers","page":"Linear solvers","title":"Linear solvers","text":"","category":"section"},{"location":"linearsolvers/","page":"Linear solvers","title":"Linear solvers","text":"A linear solver should support the solve_linear! function specified below. ","category":"page"},{"location":"linearsolvers/","page":"Linear solvers","title":"Linear solvers","text":"FESolvers.solve_linear!","category":"page"},{"location":"linearsolvers/#FESolvers.solve_linear!","page":"Linear solvers","title":"FESolvers.solve_linear!","text":"solve_linear!(Δx, K, r, linearsolver)\n\nUsing the method specified by linearsolver,  solve K Δx = -r for Δx\n\n\n\n\n\n","category":"function"},{"location":"linearsolvers/#Implemented-Solvers","page":"Linear solvers","title":"Implemented Solvers","text":"","category":"section"},{"location":"linearsolvers/#BackslashSolver","page":"Linear solvers","title":"BackslashSolver","text":"","category":"section"},{"location":"linearsolvers/","page":"Linear solvers","title":"Linear solvers","text":"BackslashSolver","category":"page"},{"location":"linearsolvers/#FESolvers.BackslashSolver","page":"Linear solvers","title":"FESolvers.BackslashSolver","text":"BackslashSolver()\n\nThe standard julia linear solver using Δx .= -K\\r\n\n\n\n\n\n","category":"type"},{"location":"userfunctions/#User-problem","page":"User problem","title":"User problem","text":"","category":"section"},{"location":"userfunctions/","page":"User problem","title":"User problem","text":"The key to using the FESolvers.jl package is to define your  problem. This problem should support a set of functions in order for the solver to solve your problem.  While some functions are always required, some are only required by certain solvers.  Furthermore, a two-level API exist: Simple and advanced.  The simple API does not expose which solver is used, while the advanced API requires you to dispatch on the type of solver. ","category":"page"},{"location":"userfunctions/#Applicable-to-all-solvers","page":"User problem","title":"Applicable to all solvers","text":"","category":"section"},{"location":"userfunctions/","page":"User problem","title":"User problem","text":"FESolvers.getunknowns\nFESolvers.getresidual\nFESolvers.update_to_next_step!\nFESolvers.update_problem!\nFESolvers.handle_converged!\nFESolvers.postprocess!","category":"page"},{"location":"userfunctions/#FESolvers.getunknowns","page":"User problem","title":"FESolvers.getunknowns","text":"getunknowns(problem)\n\nReturn the current vector of unknown values\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.getresidual","page":"User problem","title":"FESolvers.getresidual","text":"getresidual(problem)\n\nReturn the current residual vector of the problem\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.update_to_next_step!","page":"User problem","title":"FESolvers.update_to_next_step!","text":"update_to_next_step!(problem, time)\n\nUpdate prescribed values, external loads etc. for the given time.\n\nThis function is called in the beginning of each new time step.  Note that for adaptive time stepping, it may be called with a lower  time than the previous time if the solution did not converge.\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.update_problem!","page":"User problem","title":"FESolvers.update_problem!","text":"update_problem!(problem)\nupdate_problem!(problem, Δx)\n\nAssemble the residual and stiffness for x+=Δx. \n\nSome linear solvers may be inaccurate, and if modified stiffness is used  to enforce constraints on x, it is good the force Δx=0 on these components inside this function. \nΔx is not given in the first call after update_to_next_step! in which case no change of x should be made. \n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.handle_converged!","page":"User problem","title":"FESolvers.handle_converged!","text":"handle_converged!(problem)\n\nDo necessary update operations once it is known that the  problem has converged. E.g., update old values to the current.  Only called directly after the problem has converged. \n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.postprocess!","page":"User problem","title":"FESolvers.postprocess!","text":"postprocess!(problem, step)\n\nPerform any postprocessing at the current time and step nr step Called after time step converged, and after handle_converged!\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#Simple-API","page":"User problem","title":"Simple API","text":"","category":"section"},{"location":"userfunctions/","page":"User problem","title":"User problem","text":"FESolvers.calculate_convergence_measure\nFESolvers.getjacobian\nFESolvers.getdescentpreconditioner","category":"page"},{"location":"userfunctions/#FESolvers.calculate_convergence_measure","page":"User problem","title":"FESolvers.calculate_convergence_measure","text":"calculate_convergence_measure(problem) -> AbstractFloat\n\nCalculate a value to be compared with the tolerance of the nonlinear solver.  A standard case when using Ferrite.jl is norm(getresidual(problem)[Ferrite.free_dofs(dbcs)])  where dbcs::Ferrite.ConstraintHandler.\n\nThe advanced API alternative is check_convergence_criteria\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.getjacobian","page":"User problem","title":"FESolvers.getjacobian","text":"getjacobian(problem)\n\nReturn the jacobian drdx, or approximations thereof.\n\nMust be defined for NewtonSolver, but can also be  defined by the advanced API alternative getsystemmatrix:  getsystemmatrix(problem, ::NewtonSolver)\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.getdescentpreconditioner","page":"User problem","title":"FESolvers.getdescentpreconditioner","text":"getdescentpreconditioner(problem)\n\nReturn a preconditioner K for calculating the descent direction p, considering solving r(x)=0 as a minimization problem of f(x) where r=∇f. The descent direction is then p = K⁻¹ ∇f\n\nUsed by the SteepestDescent solver, and defaults to I if not defined.  The advanced API alternative is getsystemmatrix:  getsystemmatrix(problem, ::SteepestDescent)\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#Advanced-API","page":"User problem","title":"Advanced API","text":"","category":"section"},{"location":"userfunctions/","page":"User problem","title":"User problem","text":"FESolvers.getsystemmatrix\nFESolvers.check_convergence_criteria","category":"page"},{"location":"userfunctions/#FESolvers.getsystemmatrix","page":"User problem","title":"FESolvers.getsystemmatrix","text":"getsystemmatrix(problem,nlsolver)\n\nReturn the system matrix of the problem. For a Newton solver this method should return the Jacobian, while for a steepest descent method this can be a preconditioner as e.g., the L2 product of the gradients. By default the system matrix for the SteepestDescent method is the unity matrix and thus, renders a vanilla gradient descent solver.\n\n\n\n\n\n","category":"function"},{"location":"userfunctions/#FESolvers.check_convergence_criteria","page":"User problem","title":"FESolvers.check_convergence_criteria","text":"check_convergence_criteria(problem, nlsolver) -> Bool\n\nCheck if problem has converged and update the state  of nlsolver wrt. number of iterations and a convergence measure if applicable.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FESolvers","category":"page"},{"location":"#FESolvers","page":"Home","title":"FESolvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for FESolvers.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"solve_problem!\nQuasiStaticSolver","category":"page"},{"location":"#FESolvers.solve_problem!","page":"Home","title":"FESolvers.solve_problem!","text":"solve_problem!(solver, problem)\n\nSolve a given user problem using the chosen solver\n\nFor details on the functions that should be defined for problem, see User problem\n\n\n\n\n\n","category":"function"},{"location":"#FESolvers.QuasiStaticSolver","page":"Home","title":"FESolvers.QuasiStaticSolver","text":"QuasiStaticSolver(nlsolver, timestepper)\n\nA quasi-static solver that solves problems of the type R(X(t),t)=0.  It has two parts: A nonlinear solver (see Nonlinear solvers)  and a time stepper (see Time steppers). \n\n\n\n\n\n","category":"type"},{"location":"timesteppers/#Time-steppers","page":"Time steppers","title":"Time steppers","text":"","category":"section"},{"location":"timesteppers/","page":"Time steppers","title":"Time steppers","text":"A time stepper should support the following functions","category":"page"},{"location":"timesteppers/","page":"Time steppers","title":"Time steppers","text":"FESolvers.initial_time\nFESolvers.islaststep\nFESolvers.update_time","category":"page"},{"location":"timesteppers/#FESolvers.initial_time","page":"Time steppers","title":"FESolvers.initial_time","text":"initial_time(timestepper)\n\nReturn the starting time for the given timestepper\n\n\n\n\n\n","category":"function"},{"location":"timesteppers/#FESolvers.islaststep","page":"Time steppers","title":"FESolvers.islaststep","text":"islaststep(timestepper, time, step)->Bool\n\nReturn true if the current step/time is the last step, return false otherwise \n\n\n\n\n\n","category":"function"},{"location":"timesteppers/#FESolvers.update_time","page":"Time steppers","title":"FESolvers.update_time","text":"update_time(solver, time, step, converged::Bool)\nupdate_time(timestepper, nlsolver, time, step, converged::Bool)\n\nReturn the next time and step number, depending on if the previous time step converged  or not. If not converged, return the same step but a new_time<time to reduce the  time step. If it is not possible to retry with shorter timestep, throw  ConvergenceError. If converged, update time step as planned.  Note: The full solver is given as input to allow specialization on e.g. if a  Newton iteration has required many iterations, shorten the next time step as a  precausionary step.\n\nNote that a call to the first definition is forwarded to the second function definition  by decomposing the solver, unless another specialization is defined.\n\n\n\n\n\n","category":"function"},{"location":"timesteppers/#Implemented-steppers","page":"Time steppers","title":"Implemented steppers","text":"","category":"section"},{"location":"timesteppers/#FixedTimeStepper","page":"Time steppers","title":"FixedTimeStepper","text":"","category":"section"},{"location":"timesteppers/","page":"Time steppers","title":"Time steppers","text":"FixedTimeStepper","category":"page"},{"location":"timesteppers/#FESolvers.FixedTimeStepper","page":"Time steppers","title":"FESolvers.FixedTimeStepper","text":"FixedTimeStepper(num_steps::int, Δt, t_start=0)\nFixedTimeStepper(t::Vector)\n\nA time stepper which gives fixed time steps. If the convenience interface is used, constant increments are used. Note that  length(t)=num_steps+1 since the first value is just the initial  value and is not an actual step.  \n\n\n\n\n\n","category":"type"}]
}
