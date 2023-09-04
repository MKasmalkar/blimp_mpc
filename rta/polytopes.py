# File: polytopes.py
# Purpose: matrix operations for computing controllable sets
#
# See: Predictive Control for linear and hybrid systems by F. Borrelli, A. Bemporad, M. Morari (2016)
#      for most of the math. Implementation is my own.
# 
# Fouier-Mozkin is my own implementation
#
# Note that this code file works, but could be much more efficiently programmed for small discretization steps.
# Update 07/09/23: pypoman has efficient implementations of projections, which helps speed up this code file.
#
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2022

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from numpy import linalg
from pypoman import project_polytope, compute_polytope_halfspaces, compute_polytope_vertices, plot_polygon

class Polytope():
    # Polytopes are H-polytopes of the form Px <= p
    def __init__(self, *args):
        if len(args) == 1:
            assert(args[0].shape[0] == args[0].shape[0])
            self.P = args[0].P
            self.p = args[0].p
        elif len(args) == 2:
            assert(args[0].shape[0] == args[0].shape[0])
            self.P = args[0] # P is a numpy array, n x m
            self.p = args[1] # p is a numpy array, n x 1
            # thus, x is m x 1. (n constraints in m dimensions)
        else:
            self.P = np.array([[np.NaN]], dtype='float32') # n = 1, by default.
            self.p = np.array([[np.NaN]], dtype='float32')

        self.debug=False
    def __str__(self):
        return f'P:\n{self.P}\np:\n{self.p}\n'

    def plot_polytope(self, m_title='Polytopic plot', m_xlabel='x_1', m_ylabel='x_2', save=False):
        # plots self.
        p_figure, p_axis = plt.subplots()
        num_equations = self.P.shape[0]
        max_x = -np.inf
        max_y = -np.inf
        min_x = np.inf
        min_y = np.inf

        V = compute_polytope_vertices(self.P, self.p)
        try:
            plot_polygon(V, color='b')
            # p_axis.fill(V[:][0], V[:][1], 'b')
            x = np.array([nu[0] for nu in V])
            y = np.array([nu[1] for nu in V])
            max_x = np.max(x)
            max_y = np.max(y)
            min_x = np.min(x)
            min_y = np.min(y)
            p_axis.set_xlim(left=min_x-0.1, right=max_x+0.1)
            p_axis.set_ylim(bottom=min_y-0.1, top=max_y+0.1)
        except:
            for j in range(-1, len(V)-1):
                if V[j].size == 2:
                    plt.plot([V[j][0], V[j+1][0]], [V[j][1], V[j+1][1]], 'b.-')
                else:
                    plt.plot([V[j][0], V[j+1][0]], [0, 0], 'b.-')
        
        p_axis.grid(True)
        p_axis.set_title(m_title)
        p_axis.set_xlabel(m_xlabel)
        p_axis.set_ylabel(m_ylabel)
        p_figure.subplots_adjust(bottom=0.15)
        p_figure.set_figheight(3)
        if save:
            p_figure.savefig('output/controllable_set.pdf', format='pdf')

    def to_augmented_matrix(self):
        # Converts self to an augmented matrix
        return np.hstack((self.P, self.p))

    def vstack(self, X):
        # Stacks the polytope X below self.
        self.P = np.vstack((self.P, X.P))
        self.p = np.vstack((self.p, X.p))

    def check_feasibility(self, X):
        # X is a column vector in R^n.
        return np.all(self.P @ X <= self.p)

    def minrep(self):
        # MINREP Reduces an H-polytope into its minimal representation.
        # This requires solving linear programs.

        # first, deal with any rows that are invalid.
        # Check that the matrix does not contain any rows that are stuff like
        # 0x1 + 0x2 <= 0.26
        if self.debug:
            print('[DEBUG] begin minimal representation routine.')
        n = self.P.shape[0]
        entries_to_include = []
        for i in range(n):
            if np.sum(np.abs(self.P[i,:])) >= 1e-8:
                entries_to_include.append(i)
        self.P = self.P[entries_to_include, :]
        if len(self.p.shape) < 2:
            self.p = np.expand_dims(self.p[entries_to_include], axis=0)
        else:
            self.p = self.p[entries_to_include, :]

        n, m = self.P.shape

        # Re-write this like the textbook algorithm
        entries_to_include = [x for x in range(n)]
        augmentedMatrix = self.to_augmented_matrix()
        if self.debug:
            print(augmentedMatrix)
            print(augmentedMatrix.shape)
        for i in range(n):
            augmentedMatrix[i,-1] += 1 # does this increase the domain of the polytope?
            # ith constraint: augmentedMatrix[i, 0:-1] x <= augmentedMatrix[i, -1]

            # if np.any(
            #     np.all(
            #         np.abs(
            #             augmentedMatrix[i, :] - np.hstack((self.P[entries_to_include, :], self.p[entries_to_include, :]+1))
            #             ) < 1e-6, axis=1
            #         )
            #     ):
            #     continue # This means that we have a duplicate, to 1e-6 precision.

            c = -1 * augmentedMatrix[i:i+1, 0:-1].T # min cTx s.t. a bunch of things.
            # Note that we multiply this by -1, to convert to a minimization problem.

            # x0 = np.zeros(m, 1)

            # f = max_x aM(i, 1:end-1) x s.t. aM(i, 1:end-1) x < aM(i, end);
            # or: multiply by -1 and use min.

            res = linprog(c, A_ub=augmentedMatrix[entries_to_include, 0:-1], b_ub=augmentedMatrix[entries_to_include, -1:], bounds=(None,None))
            # This is a linear program. Solve with scipy's built-in minimization function.
            f = -res.fun # ideally this self.p[i]. The negative sign is for minimization.
            
            if (f > self.p[i] - 1e-6): # the cost did not change - we need this constraint
                # duplicate
                # entries_to_include.append(i)
                pass
                # Do nothing
            else: # the cost changed - this constraint is redundant.
                entries_to_include.remove(i)
                # print('redundant')
            augmentedMatrix[i, -1] -= 1 # restore what we added.
            
        self.P = self.P[entries_to_include, :]
        if len(self.p.shape) < 2:
            self.p = np.expand_dims(self.p[entries_to_include], axis=0)
        else:
            self.p = self.p[entries_to_include, :]
        if self.debug:
            print('[DEBUG] end minimal representation routine.')
            print(entries_to_include)

class MatrixMath():
    def __init__(self):
        pass
    def precursor(A, B, Hx, Hu):
        # Calculate the precursor set, given a state polytope Hx and control polytope Hu
        # This function calculates a 1-step precursor set, for now.

        # Calculate a static matrix for filling in the '0' in the predecessor set matrix calculation
        Z = np.zeros((Hu.P.shape[0], Hx.P.shape[1]))
        
        ctrlX_temp = np.vstack((
            np.hstack((Hx.P @ A, Hx.P @ B)),
            np.hstack((Z, Hu.P))
        ))
        ctrlx_temp = np.vstack((Hx.p, Hu.p))

        n = ctrlX_temp.shape[1] # dimension of the original polytope
        p = A.shape[0] # dimension of the projected polytope
        ineq = (ctrlX_temp, ctrlx_temp)
        E = np.zeros((p, n))
        for j in range(p):
            E[j, j] = 1
        f = np.zeros((p,))
        proj = (E,f)
        vertices = project_polytope(proj, ineq=ineq)
        halfspaces = compute_polytope_halfspaces(vertices)
        return Polytope(halfspaces[0], np.expand_dims(halfspaces[1], axis=1))

    def intersect(poly1, poly2):
        # Intersect two polytopes
        # Stack the polytopes
        P = Polytope(np.vstack((poly1.P, poly2.P)), np.vstack((poly1.p, poly2.p)))
        return P

    def controllable_set(A, B, dT, Hx, Hu):
        assert(A.shape[0] == Hx.P.shape[1])
        assert(B.shape[1] == Hu.P.shape[1])
        # A, B - system dynamics
        # dT - sampling rate
        # Hx, Hu - system constraints as polytopes.
        n = round(2 / dT)

        # Calculate a static matrix for filling in the '0' in the predecessor set matrix calculation
        Z = np.zeros((Hu.P.shape[0], Hx.P.shape[1]))

        for i in range(n):
            # Calculate the predecessor set.
            ctrlX_temp = np.vstack((
                np.hstack((Hx.P @ A, Hx.P @ B)),
                np.hstack((Z, Hu.P))
            ))
            ctrlx_temp = np.vstack((Hx.p, Hu.p))

            n = ctrlX_temp.shape[1] # dimension of the original polytope
            p = A.shape[0] # dimension of the projected polytope

            ineq = (ctrlX_temp, ctrlx_temp)

            E = np.zeros((p, n))
            for j in range(p):
                E[j, j] = 1
            f = np.zeros((p,))
            proj = (E,f)
            vertices = project_polytope(proj, ineq=ineq)
            halfspaces = compute_polytope_halfspaces(vertices)
            Hx.vstack(Polytope(halfspaces[0], np.expand_dims(halfspaces[1], axis=1)))

        return Hx
    def minkowski_sum(polytope1, polytope2):
        # Create two large matrices.
        Z = np.zeros((polytope1.P.shape[0], polytope2.P.shape[1]))
        L = np.vstack((np.hstack((Z, polytope1.P)), np.hstack((polytope2.P, -polytope2.P))))
        l = np.vstack((polytope1.p, polytope2.p))

        print('in Minkowski sum')

        n = L.shape[1] # the number of inequalities
        p = polytope1.P.shape[1]

        ineq = (L, l)
        E = np.zeros((p, n))
        for j in range(p):
            E[j, j] = 1
        f = np.zeros((p,))
        proj = (E, f)
        print(f'projecting... {p}, {n}')
        vertices = project_polytope(proj, ineq=ineq)
        print('halfspaces...')
        halfspaces = compute_polytope_halfspaces(vertices)
        # if self.debug:
        # print(halfspaces)
        # print(np.expand_dims(halfspaces[1], axis=1))
        return (Polytope(halfspaces[0], np.expand_dims(halfspaces[1], axis=1)), True)

        # return MatrixMath.project(Polytope(L, l), polytope1.P.shape[1])
    def linear_mapping(polytope1, A=None, b=None):
        # Applies a linear mapping to an H-polytope
        # H = P / A
        H = Polytope()
        if A is None:
            A = np.eye(polytope1.P.shape[0])
        if b is None:
            b = np.zeros((A.shape[0], 1))
        try:
            H.P = polytope1.P @ linalg.inv(A)
            H.p = polytope1.p + H.P @ b
            return H
        except:
            # Linear algebra error.

            # Convert H to a V-representation
            V = compute_polytope_vertices(polytope1.P, polytope1.p)
            # Apply A to each of the entries
            V = [A@v for v in V]
            halfspaces = compute_polytope_halfspaces(V)
            return Polytope(halfspaces[0], np.expand_dims(halfspaces[1], axis=1))
    def project(polytope1, d):
        # Projects polytope1 into d dimensions
        # This function greedily implements the Fourier-Motzkin algorithm

        valid = True # default.

        n_start = polytope1.P.shape[1]

        for run_index in range(n_start - d):
            # Get the fundamental dimension of polytope1.
            n = polytope1.P.shape[0] # number of constraints
            m = polytope1.P.shape[1] # current number of dimensions

            # Project such that x[-1] = 0
            # Fourier-Motzkin: take all pairs of inequalities with opposite sign
            # coefficients of x[-1], and for each generate a new valid inequality
            # that eliminates x[-1]
            lp = polytope1.P[:, -1] > 0
            ln = polytope1.P[:, -1] < 0
            le = polytope1.P[:, -1] == 0
            list_of_positives = np.hstack((polytope1.P[lp, :], polytope1.p[lp]))
            list_of_negatives = np.hstack((polytope1.P[ln, :], polytope1.p[ln]))
            list_of_equals = np.hstack((polytope1.P[le, :], polytope1.p[le]))

            # Calculate how large inequalities needs to be
            ineq_rows = list_of_positives.shape[0] * \
                        list_of_negatives.shape[0] + \
                        list_of_equals.shape[0]
            inequalities = np.zeros((ineq_rows, m + 1))
            j = 0 # Initialize this index.
            for p_index in range(list_of_positives.shape[0]):
                for n_index in range(list_of_negatives.shape[0]):
                    # Goal: eliminate the last value in x.
                    #if np.abs(list_of_negatives[n_index, -1]) > 1e-6:
                    Lambda = list_of_positives[p_index, -2] / np.abs(list_of_negatives[n_index, -2])
                    inequalities[j,:] = list_of_positives[p_index, :] + Lambda * list_of_negatives[n_index, :]
                    j += 1

            # step 2: propogate all inequalities that don't rely on x[-1]

            for e_index in range(list_of_equals.shape[0]):
                inequalities[j,:] = list_of_equals[e_index, :]
                j += 1

            # decode inequalities back into two matrices
            polytope1.P = inequalities[:, 0:-2]
            polytope1.p = np.expand_dims(inequalities[:, -1], axis=1)
            
            # Remove zero constraints (that is, [0 0 ... 0] <= 0)
            max_iterations = polytope1.P.shape[0] # n, but new.
            true_index = 0
            for r_index in range(max_iterations):
                if true_index > polytope1.P.shape[0]:
                    break
                if np.all(polytope1.P[true_index, :] == 0):
                    if polytope1.p[true_index] < 0:
                        valid = False # we have an inconsistent formula
                    else:
                        # Remove this redundant constraint.
                        t_size = polytope1.P.shape[0]
                        polytope1.P = np.delete(polytope1.P, true_index, axis=0)
                        polytope1.p = np.delete(polytope1.p, true_index, axis=0)
                else:
                    true_index += 1 # Because this isn't C, parallelization and stuff

            # polytope1.minrep() # This didn't help. We should be move clever about removing
            # redundant constraints as we do Fourier elimination, however. See Wikipedia.
            # (c) If at any stage the matrix M of step (a) has two rows r and t 
            # satisfying m(r,j) = 0 => m(t,j) = 0, then row r can be dropped from M. 
            # (d) If at any stage a row of the matrix M of (a) has more positive 
            # entries than the numbers of variables actively eliminated plus one, then 
            # the row can be dropped from M.
            # The matrix M is the "history" - each elimination is a matrix operation.
            # We are doing a single operation with a single variable here, but we can use
            # condition (c) to remove unnecessary constraints. Might be the same as the
            # Imbert acceleration conditions?
        return polytope1, valid
