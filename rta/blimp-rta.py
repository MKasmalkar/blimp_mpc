# File: blimp-rta.py
# Purpose: Implements the algorithm in "Guaranteeing Signal Temporal Logic Safety Specifications with Runtime Assurance"
# 
# This demonstrates optimization for the blimp with a nontrivial safety specification and the nonlinear
# blimp model. The "recursively feasible set" is now a bit more complicated and requires more future time
# steps to compute it (b > 1). 
# 
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2023

import numpy as np
import time
import scipy
import scipy.io # For matlab logging.
from stlpy.systems import LinearSystem
from stlpy.STL import LinearPredicate
from stlpy.solvers import GurobiMICPWarmStartSolver
from matplotlib import pyplot as plt # Python plotting
from matplotlib import rc
from polytopes import Polytope
from blimp import Blimp

# Function: persistently_feasible_set(A,B)
# Purpose: compute a polytope as a terminal set constraint
def persistently_feasible_set(A,B,u_max, square_dim):
    # Create two polytopes: one representing bounds on x, the other representing bounds on u.

    # H_x = np.array([[1,0], [-1, 0], [0, 1], [0, -1]])
    # h_x = np.array([[1.1], [-0.9], [1], [1]]) # velocity bounds are an initial overestimate.
    H_x = np.zeros((17,12))
    h_x = np.zeros((17,1))
    H_x[0, 2] = 1; h_x[0, 0] = 0.1#0.5 # v_z <= 0.1
    H_x[1, 2] = -1; h_x[1, 0] = 0.1#0.5 # -v_z <= 0.1, v_z >= -0.1
    H_x[2, 8] = 1; h_x[2, 0] = 1 # z <= 1
    H_x[3, 8] = -1; h_x[3, 0] = 1 # -z <= 1, z >= -1

    H_x[4, 9] = 1; h_x[4, 0] = 0.2 # pitch <= 0.2 rad
    H_x[5, 9] = -1; h_x[5, 0] = 0.2 # pitch >= -0.2 rad
    H_x[6, 10] = 1; h_x[6, 0] = 0.2 # roll <= 0.2 rad
    H_x[7, 10] = -1; h_x[7, 0] = 0.2 # roll >= -0.2 rad

    H_x[8,4] = 1; h_x[8,0] = 0.3 # w_y <= 0.4
    H_x[9,4] = -1; h_x[9,0] = 0.3 # w_y >= -0.4
    H_x[10,3] = 1; h_x[10,0] = 0.2 # w_x <= 0.2
    H_x[11,3] = -1; h_x[11,0] = 0.2 # w_x >= -0.2

    H_x[12,5] = 1; h_x[12,0] = 0.05 # w_z <= 0.05
    H_x[13,5] = -1; h_x[13,0] = 0.05 # w_z >= -0.05
    H_x[14,11] = 1; h_x[14,0] = 10; # yaw <= 10 rad
    H_x[15,11] = -1; h_x[15,0] = 10; # yaw >= -10 rad

    # H_u = np.array([[1], [-1]])
    # h_u = np.array([[u_max], [u_max]])
    # H_u = np.ones((4,8))
    H_u = np.block([[np.eye(4)],[np.eye(4)*-1]])
    # print(H_u)
    # H_u[0, :] = 1
    # H_u[1, :] = -1
    # h_u = np.ones((8,1)) * u_max # TODO uncomment.
    Hx = Polytope(H_x, h_x)
    # Hu = Polytope(H_u, h_u) # TODO uncomment.
    # controllable_set_start = time.time()
    # Qp = MatrixMath.controllable_set(A, B, dT, Hx, Hu)
    # print("--- Controllable set computation took %s seconds ---" % (time.time() - controllable_set_start))
    # Qp.plot_polytope(m_title=f'Maximal control invariant set, $\Delta t={dT}$', m_xlabel='$x_1$', m_ylabel='$x_2$', save=True)

    # I need to create five persistently feasible sets as follows:
    # Set (1): convex constraints.
    # Set (j), non-convex constraints OR'd.
    H_x_left = np.zeros((2, 12))
    h_x_left = np.zeros((2, 1))
    H_x_left[0, 6] = 1; h_x_left[0, 0] = -square_dim # x <= -square_dim
    H_x_left[1, 0] = 1; h_x_left[1, 0] = 0 # v_x <= 0

    H_x_right = np.zeros((2, 12))
    h_x_right = np.zeros((2,1))
    H_x_right[0, 6] = -1; h_x_right[0, 0] = -square_dim # -x <= -square_dim, x >= square_dim
    H_x_right[1, 0] = -1; h_x_right[1, 0] = 0; # -v_x <= 0

    H_y_top = np.zeros((2, 12))
    h_y_top = np.zeros((2, 1))
    H_y_top[0, 7] = -1; h_y_top[0, 0] = -square_dim
    H_y_top[1, 1] = -1; h_y_top[1, 0] = 0

    H_y_bottom = np.zeros((2, 12))
    h_y_bottom = np.zeros((2, 1))
    H_y_bottom[0, 7] = 1; h_y_bottom[0, 0] = -square_dim
    H_y_bottom[1, 0] = 1; h_y_bottom[1, 0] = 0

    # For now:
    return (Hx, Polytope(H_x_left, h_x_left), Polytope(H_x_right, h_x_right), Polytope(H_y_top, h_y_top), Polytope(H_y_bottom, h_y_bottom))

def main():
    rc('text', usetex=True)

    dT = 0.25

    square_dim = 3 # m, the dimensions of the square.

    x0 = np.zeros((12,1)) # start at the origin
    x0[7, 0] = -1 # OK, place y at y=-1

    # Simulation parameters.
    sim_length_seconds = 120 # in seconds
    t_np = np.arange(0, sim_length_seconds, dT)
    sim_length = t_np.shape[0]

    # Nominal input.
    u_nominal = np.zeros((4, sim_length)) * dT # uncomment for constant input, first & second runs
    u_nominal[0,:] = 0.03
    # u_nominal[3,:] = 0.5
    print('nominal input vector')
    print(u_nominal)

    # Set maximum input, i.e. the polytope U.
    u_constraints = np.array([[-0.0672, 0.0672],
                      [-0.0672, 0.0672],
                      [-0.0366, 0.0908],
                      [-0.0013, 0.0013]])
    
    # Construct the blimp model.
    my_blimp = Blimp()
    my_blimp.setup_discrete_lti_model(dT)
    my_blimp.discretize(dT=dT)
    my_blimp.compute_stable_physics_model_feedback()
    
    # Extract matrices for their size.
    A,B = my_blimp.stable_physics_model()

    # Create and set the outputs of the blimp in R6.
    C = np.block([np.zeros((6,6)), np.eye(6)]) # outputs = positions and angles.
    my_blimp.C_(C)

    # Current blimp model:
    # x[k+1] = (A - BK)x[k] + Bu[k]
    # y[k] = C x[k]

    # Build the STL Formulae. "Positive Normal Form" (can only negate predicates.)
    A_height = np.zeros((6,1))
    A_height[2, 0] = 1
    lb_h = LinearPredicate(A_height, -0.5) # above z=-0.5
    ub_h = LinearPredicate(-A_height, -0.5) # below z=0.5
    # Lower and upper bounds on the z-axis
    height_spec = lb_h & ub_h # always required.

    A_pitch = np.zeros((1,6))
    A_roll = np.zeros((1,6))
    A_pitch[0,4] = 1#0.5
    A_roll[0,3] = 1
    lb_theta = LinearPredicate(A_pitch, -0.2) # above -0.2
    ub_theta = LinearPredicate(-A_pitch, -0.2) # below 0.2
    lb_phi = LinearPredicate(A_roll, -0.2)
    ub_phi = LinearPredicate(-A_roll, -0.2)
    angle_spec = (lb_theta & ub_theta & lb_phi & ub_phi).always(0,2) # always required

    # Create the square.
    A_x = np.zeros((1,6))
    A_y = np.zeros((1,6))
    A_x[0, 0] = 1
    A_y[0, 1] = 1
    lb_x = LinearPredicate(-A_x, square_dim) # Safe: x <= -square_dim
    ub_x = LinearPredicate(A_x, square_dim) # Safe: x >= square_dim
    lb_y = LinearPredicate(-A_y, square_dim) # Safe: y <= -square_dim
    ub_y = LinearPredicate(A_y, square_dim) # Safe: y >= square_dim
    x_between_0_and_20_spec = (lb_x | ub_x | lb_y | ub_y) | (lb_x | ub_x | lb_y | ub_y).always(0, round(3/dT) - 1).eventually(0, round(3/dT) - 1)

    pi = (x_between_0_and_20_spec, angle_spec, height_spec) # 

    # Recall: implication a=>b is ~a | b.
    
    # Setup model parameters.
    horizon = round(3 / dT) * 2
    fts = horizon*2 # future time steps to project the system out. (N=horizon, b=1, applying Proposition 1 in the paper).
    N = horizon-1 #int(horizon / 2) # Where to place the terminal set constraint along the prediction horizon.

    # Create data structures to save past states and inputs.
    x_hist = np.zeros((12, sim_length))
    x_hist_continuous = np.zeros((12, sim_length * int(dT / 0.0001)))
    x_hist_continuous[:, 0:1] = x0
    x_hist[:, 0:1] = x0
    u_hist = np.zeros((4, sim_length))
    rho_hist = np.zeros((1, sim_length)) + np.NaN
    cost_hist = np.zeros((1, sim_length))

    # Create a dummy system with the same dimensions as the blimp to avoid errors.
    D = np.zeros((6,4))
    sys = LinearSystem(A, B, C, D)

    # Logging: system dimensions.
    print('System dimensions.')
    print(f'p: {sys.p}, n: {sys.n}, m: {sys.m}')

    # Construct the erminal set constraint.
    Qp = persistently_feasible_set(A,B,u_constraints, square_dim)

    # Construct a hard constraint solver without any slack
    # solver = GurobiMICPWarmStartSolver(spec=pi, sys=sys, model=my_blimp, x0=x_hist[:, 0:1], T=horizon+fts,
    #          robustness_cost=False, horizon=horizon, verbose=False, dT=dT, N=N)
    # solver.AddControlBounds(-u_max, u_max)
    # solver.AddRecursiveFeasibilityConstraint(Qp)
    # solver.AddLPCost() # Required for LP formulation.

    # Construct a backup solver with slack, in case the first solver ever becomes infeasible.
    backup_solver = GurobiMICPWarmStartSolver(spec=pi, sys=sys, model=my_blimp, x0=x_hist[:, 0:1], T=horizon+fts,
             robustness_cost=True, hard_constraint=False, horizon=horizon, verbose=False, dT=dT, N=N, rho_min=-np.inf) # N=horizon)
    backup_solver.AddControlBounds(u_constraints[:, 0], u_constraints[:, 1])
    backup_solver.AddRecursiveFeasibilityConstraint(Qp)
    backup_solver.AddLPCost()

    start_time = time.time() # to measure execution time.
    u_last = np.zeros((4,1)) # for use with the softened optimization problem.
    x_last = None            # for use with the softened optimization problem.

    # Nominal controller parameters
    Kd = 0.22
    Kp = 0.3
    targets = np.array([[0, 3.5], [3.5, 0], [0, -3.5], [-3.5, 0]])
    close_enough = 0.05 # within this, switch to the next waypoint for the blimp
    current_target = 2 # current_target = 0, 1, 2, or 3

    for i in range(1, sim_length):
        # Update the nominal input.
        # Setup (4) waypoints, and use a PD controller to approach each of them.

        currentXY = x_hist[6:8, i-1]
        print(currentXY)
        if currentXY[0] - close_enough <= targets[current_target, 0] and currentXY[0] + close_enough >= targets[current_target, 0] \
            and currentXY[1] - close_enough <= targets[current_target, 1] and currentXY[1] + close_enough >= targets[current_target, 1]:
            # print(f'current target: {targets[current_target]}')
            # print("close enough!")
            current_target = (current_target + 1) % 4

        currentV_XY = x_hist[0:2, i-1]
        # print(-Kp * (currentXY - targets[current_target,:]) - Kd * currentV_XY)
        u_nominal[0:2, i-1] = -Kp * (currentXY - targets[current_target,:]) - Kd * currentV_XY
        # print(u_nominal[:, i-1:i])
        # End update of the nominal input.

        backup_solver.updateModel(sys=sys, x0=x_hist[:, 0:i], i=i, u_hat=u_nominal[:, i-1:i],
                                    initialization_point=x_last) 
        solution = backup_solver.Solve()
        x_1 = solution[0]
        u_last = solution[1]
        rho_1 = solution[2]
        # print(f'rho is {solution[2]}')
        obj_1 = solution[4]
        try:
            x_last = x_1[:, horizon:] # for an LTV model, in the future

            # Update the historical robustness
            if i >= horizon:
                rho_hist[:, i-horizon:i-horizon+1] = rho_1[0, :, 0:1] # 0 is the safety specification

            # Old propagation.
            # x_hist[:, i:i+1] = x_1[:, horizon:horizon+1]
            # u_hist[:, i-1] = u_last[:, horizon-1]
            # continue

            # New propagation.
            x_next = x_hist[:, i-1:i].copy() # get the previous state.
            u_next = u_last[:, horizon-1:horizon] # get the computed input, and apply it as a zero-order hold.
            for __ in range(int(dT / 0.0001)): # Propagate the state with a time step of 0.0001
                x_next += 0.0001 * my_blimp.f(x_next, u_next) # 0.0001 *
                x_hist_continuous[:, i * int(dT / 0.0001) + __: i * int(dT / 0.0001) + __ + 1] = x_next.copy()

                # my_blimp.f(x_next, u_next)

            # Update the saved historical state / input
            x_hist[:, i:i+1] = x_next.copy() # Propagated based on a time step of 0.0001 instead of dT
            u_hist[:, i-1:i] = u_next

            # Update the historical cost
            cost_hist[0, i] = np.linalg.norm(u_next - u_nominal[:, i-1:i], 1)

        except TypeError as e:
            print("A TypeError occurred.")
            print(e)
            break

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Export the results for matlab.
    scipy.io.savemat('blimp-rta-output.mat', mdict={'x': x_hist_continuous, 'u': u_hist, 't': t_np})

    basic_line = np.ones((sim_length)) # for plotting "y in bounds"

    # # uncomment to print the results as individual plots.
    # position (x_1)
    # x_figure, x_axis = plt.subplots()
    # x_axis.plot(t_np, x_hist[0, 0:sim_length], 'b.-')
    # x_axis.plot(t_np, basic_line * 1.1, 'r')
    # x_axis.plot(t_np, basic_line * 0.9, 'r')
    # x_axis.grid(True)
    # x_axis.set_title('Output $y[t]$')
    # x_axis.set_xlabel('$t$ (s)')
    # x_axis.set_ylabel('$y[t]$')
    # x_figure.set_figheight(4)
    # x_figure.savefig('output/x.pdf', format='pdf')

    # # velocity (x_2)
    # v_figure, v_axis = plt.subplots()
    # v_axis.plot(t_np, x_hist[1, 0:sim_length], 'b.-')
    # v_axis.grid(True)
    # v_axis.set_title('$x_2[t]$')
    # v_axis.set_xlabel('t (s)')
    # v_axis.set_ylabel('$x_2[t]$')
    # v_figure.set_figheight(4)
    # v_figure.savefig('output/v.pdf', format='pdf')

    # input
    u_figure, u_axis = plt.subplots()
    i_u_1, = u_axis.plot(t_np, u_hist[0, 0:sim_length], 'b.-', alpha=0.6)
    i_u_2, = u_axis.plot(t_np, u_hist[1, 0:sim_length], 'r-', alpha=0.6)
    i_u_3, = u_axis.plot(t_np, u_hist[2, 0:sim_length], 'm-', alpha=0.6)
    i_u_4, = u_axis.plot(t_np, u_hist[3, 0:sim_length], 'k-', alpha=0.6)
    u_axis.grid(True)
    u_axis.set_title('Input $u[t]$')
    u_axis.set_xlabel('$t$ (s)')
    u_axis.set_ylabel('$u[t]$')
    u_axis.legend([i_u_1, i_u_2, i_u_3, i_u_4], ['f_x', 'f_y', 'f_z', 'T_z'], loc="lower right")
    u_figure.set_figheight(6)
    u_figure.savefig('output/u.pdf', format='pdf')

    my_blimp.plot_results(t_np, x_hist, show=False)

    main_figure, ((x_axis, u_axis), (rho_axis, cost_axis)) = plt.subplots(2,2)

    i_x_1, = x_axis.plot(t_np, x_hist[6, 0:sim_length], 'b.-')
    # i_x_2, = x_axis.plot(t_np, basic_line * 10, 'r')
    x_axis.grid(True)
    x_axis.set_title('Output $x[t]$')
    x_axis.set_xlabel('$t$ (s)')
    x_axis.set_ylabel('$x[t]$')
    x_axis.legend([i_x_1], ['A safe trajectory'], loc="lower right") # $p$ in bounds # i_x_2

    i_u_1, = u_axis.plot(t_np, u_hist[0, 0:sim_length], 'b.-', alpha=0.7)
    i_u_2, = u_axis.plot(t_np, u_nominal[0, 0:sim_length], 'k-', alpha=0.8)
    u_axis.grid(True)
    u_axis.set_title('Input $u[t]$')
    u_axis.set_xlabel('$t$ (s)')
    u_axis.set_ylabel('$u[t]$')
    u_axis.legend([i_u_1, i_u_2], ['Filtered', 'Nominal'], loc="lower right")

    i_rho_1 = rho_axis.plot(t_np, rho_hist[0, 0:sim_length], 'm.-')
    rho_axis.legend([i_rho_1], 'Robustness')
    rho_axis.grid(True)
    rho_axis.set_title('Robustness $\\rho[t]$')
    rho_axis.set_xlabel('$t$ (s)')
    rho_axis.set_ylabel('$\\rho[t]$')

    i_cost_1 = cost_axis.plot(t_np, cost_hist[0, 0:sim_length], 'm.-')
    cost_axis.legend([i_cost_1], 'Cost')
    cost_axis.grid(True)
    cost_axis.set_title('Cost $J(x,u)[t]$')
    cost_axis.set_xlabel('$t$ (s)')
    cost_axis.set_ylabel('$J(x,u)[t]$')

    main_figure.subplots_adjust(hspace=0.5)
    main_figure.set_figheight(8)

    main_figure.savefig('output/u_and_x.pdf', format='pdf')

    plt.show()
main()
# import cProfile
# import pstats
# from pstats import SortKey
# p = pstats.Stats('restats')
# p = cProfile.run('main()')
# p.sort_stats(SortKey.CUMULATIVE).print_stats()
