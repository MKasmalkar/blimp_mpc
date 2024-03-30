import gurobipy as gp
from gurobipy import GRB
import sys
import csv
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print("Please run with data file name as argument.")
    sys.exit(1)

filename = sys.argv[1]
with open("logs/" + filename, mode='r') as file:
    csv_file = csv.reader(file)
    data = []

    k = 0
    for line in csv_file:
        if k == 0:
            k = 1
            continue
        
        data.append([float(x) for x in line])

    data_np = np.array(data)
    time = data_np[:, 0]
    state = data_np[:, 1:13]
    state_dot = data_np[:, 13:25]
    ctrl = data_np[:, 25:29]
    dT = data_np[0, 34]

    env = gp.Env(empty=True)
    # env.setParam('OutputFlag', 0)
    # env.setParam('LogToConsole', 0)
    env.start()

    m = gp.Model(env=env)

    A = m.addMVar(shape=(12,12), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='A')
    
    B = np.array([[ 2.16684724,  0.,          0.,          0.,        ],
                  [ 0.,          2.16684724,  0.,          0.,        ],
                  [ 0.,          0.,          1.33351113,  0.,        ],
                  [ 0.,         -0.07378457,  0.,          0.,        ],
                  [ 0.07378457,  0.,          0.,          0.,        ],
                  [ 0.,          0.,          0.,          1.71791788,],
                  [ 0.,          0.,          0.,          0.,        ],
                  [ 0.,          0.,          0.,          0.,        ],
                  [ 0.,          0.,          0.,          0.,        ],
                  [ 0.,          0.,          0.,          0.,        ],
                  [ 0.,          0.,          0.,          0.,        ],
                  [ 0.,          0.,          0.,          0.,        ]])

    obj = 0
    for k in range(len(time) - 1):

        x = state[k, :]
        u = ctrl[k, :]

        x_next = x + dT * (A @ x + B @ u)
        x_next_T = x.T + dT * (x.T @ A.T + u.T @ B.T)
        
        error = state[k+1, :] - x_next
        errorT = state[k+1, :] - x_next_T
        
        obj += errorT @ error

    m.setObjective(obj)
    m.update()

    m.optimize()

    A_sol = A.X

    pd.options.display.float_format = '{:.3f}'.format
    #print(pd.DataFrame(A_sol))
    print(repr(A_sol))