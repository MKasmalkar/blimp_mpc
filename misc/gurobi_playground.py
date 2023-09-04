import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

def standard():
    # Create a new model
    m = gp.Model("MPC")

    A = sp.csc_matrix([
        [1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0. ],
        [0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0. ],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0. ],
        [0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0. ],
        [0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0. ],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0. ],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0. ],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0. ],
        [0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0. ],
        [0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0. ],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846]
    ])

    B = sp.csc_matrix([
        [0., -0.0726, 0., 0.0726],
        [-0.0726, 0., 0.0726, 0. ],
        [-0.0152, 0.0152, -0.0152, 0.0152],
        [-0., -0.0006, -0., 0.0006],
        [0.0006, 0., -0.0006, 0.0000],
        [0.0106, 0.0106, 0.0106, 0.0106],
        [0, -1.4512, 0., 1.4512],
        [-1.4512, 0., 1.4512, 0. ],
        [-0.3049, 0.3049, -0.3049, 0.3049],
        [-0., -0.0236, 0., 0.0236],
        [0.0236, 0., -0.0236, 0. ],
        [0.2107, 0.2107, 0.2107, 0.2107]
    ])

    Q = np.identity(12)
    R = np.identity(4)

    xmin = -5
    xmax = 5
    umin = -5
    umax = 5
    N = 10

    xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    xmin = np.tile(xmin, (N+1,1))
    xmax = np.tile(xmax, (N+1,1))
    umin = np.tile(umin, (N,1))
    umax = np.tile(umax, (N,1))

    x = m.addMVar(shape=(N+1,12), lb=xmin, ub=xmax, name='x')
    z = m.addMVar(shape=(N+1,12), lb=-GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N,4), lb=umin, ub=umax, name='u')

    m.addConstr(x[0, :] == np.zeros(12))
    for k in range(N):
        m.addConstr(z[k, :] == x[k, :] - xr)
        m.addConstr(x[k+1, :] == A @ x[k, :] + B @ u[k, :])
    m.addConstr(z[N, :] == x[N, :] - xr)

    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.optimize()

    print(u.X)


A = np.matrix([
    [1, 1],
    [0, 1]
])

B = np.matrix([
    [0],
    [1]
])

Q = np.identity(2)
R = np.matrix([10])

def modified(state):

    global A, B, Q, R

    # Create a new model
    m = gp.Model("MPC")

    N = 3

    order = 2

    xr = np.array([0.,0.])

    xmin = np.tile(-5, (N+1,1))
    xmax = np.tile(5, (N+1,1))
    umin = np.tile(-0.5, (N,1))
    umax = np.tile(0.5, (N,1))
    
    current_state = state.T

    x = m.addMVar(shape=(N+1,order), lb=xmin, ub=xmax, name='x')
    z = m.addMVar(shape=(N+1,order), lb=-GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N,1), lb=umin, ub=umax, name='u')

    m.addConstr(x[0, :] == current_state)

    for k in range(N):
        m.addConstr(z[k, :] == x[k, :] - xr)
        m.addConstr(x[k+1, :] == A @ x[k, :] + B @ u[k, :])
    m.addConstr(z[N, :] == x[N, :] - xr)

    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.optimize()

    return u.X[0]

def mine():
    N = 3
    order = 2

    current_state = np.array([[-4.5],
                            [2]]).T

    A = np.array([[1, 1],
                [0, 1]])

    B = np.array([[0],
                [1]])

    Q = np.identity(2)
    R = np.matrix([10])

    m = gp.Model("MPC")

    xmin = np.tile(-5, (N+1, 1))
    xmax = np.tile(5, (N+1, 1))
    umin = np.tile(-0.5, (N, 1))
    umax = np.tile(0.5, (N, 1))

    # rows = time steps
    # columns = state/input values
    x = m.addMVar(shape=(N+1, order),
                    lb=xmin, ub=xmax, name='x')
    z = m.addMVar(shape=(N+1, order),
                    lb=-GRB.INFINITY, name='z')
    u = m.addMVar(shape=(N, 1),
                    lb=umin, ub=umax, name='u')

    m.addConstr(x[0, :] == current_state)

    xr = np.array([0, 0])

    # add running cost
    for k in range(N):
        m.addConstr(z[k, :] == x[k, :])
        m.addConstr(x[k+1, :] == A @ x[k, :] + B @ u[k, :])

    obj1 = sum(z[k, :] @ Q @ z[k, :] for k in range(N+1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.optimize()

    print(u.X)
    print(x.X)


tt = np.arange(0, 30, 0.1)
x = np.matrix([[-4.5],
               [2]])

for t in tt:
    print("Time: " + str(t))
    x_old = x.copy()
    u = modified(x)[0]

    print("Old state: " + str(x_old.T))
    print("Input: " + str(u.T))
    x = (A @ x) + (B * u)
    print("New state: " + str(x.T))

    print()