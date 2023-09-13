import numpy as np

def mm(A, B):
    if not isinstance(A[0], list):
        A = [[A[i]] for i in range(len(A))]
    if not isinstance(B[0], list):
        B = [[B[i] for i in range(len(B))]]

    res = [[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*B)]
                                for A_row in A]
    
    if len(res) == 1 and not isinstance(res[0], list):
        return float(res[0])
    else:
        return list(res)

def mm3(A, B, C):
    return mm(mm(A, B), C)

def T(A):
    if isinstance(A, list) and isinstance(A[0], list):
        A_T = np.zeros((len(A[0]), len(A)))
        for row in range(len(A)):
            for col in range(len(A[row])):
                A_T[col][row] = A[row][col]

        return A_T

    elif isinstance(A, list):
        return list([[A[i]] for i in range(len(A))])
    else:
        return A