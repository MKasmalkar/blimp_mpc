# Luke Baird
# (c) Georgia Institute of Technology, 2023
# Code for analyzing classical control properties of the blimp, including stability, controllability, stabilizability, etc.
import numpy as np
from blimp import Blimp
import matplotlib.pyplot as plt
from scipy.signal import place_poles
import scipy

def plot_the_poles_of(A, title, unit_circle=False):
    # Plot the poles of matrix A in the complex plane.
    w,v = np.linalg.eig(A) # w = eigenvalues, v = right eigenvectors.
    pole_figure, pole_plot = plt.subplots()
    for pole in w:
        pole_plot.plot(np.real(pole), np.imag(pole), 'bx')
    if unit_circle:
        # plot a unit circle.
        theta = np.linspace(0, 2*np.pi, 150)
        radius = 1.0
        a = radius * np.cos(theta)
        b = radius * np.sin(theta)
        pole_plot.plot(a, b, 'k')
        pole_plot.set_aspect(1)
    pole_plot.grid(True)
    pole_plot.set_title(title)
    plt.show()

def main():
    # Blimp dynamics
    # f : R^12 x R^4 --> R^12.
    dT = 0.25#0.25
    my_blimp = Blimp()
    # my_blimp.discretize(dT=dT)

    # A, B = my_blimp.simplified_model_2(x0=np.zeros((12,1)), steps=1)
    A = my_blimp.jacobian_np(np.zeros((12,1)))
    B = my_blimp.B
    # print(A)
    # print(B)
    w,v = np.linalg.eig(A)

    # A = np.eye(12) + dT * A
    # B = my_blimp.B
    # x[t+1] = Ax[t] + Bu[t]

#     K = np.array([[-0.9358, -0.0000, -0.0000, -0.0000, 0.3077, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3790, 0.0000, ],
# [-0.0000, -0.9358, 0.0000, 0.3077, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3790, 0.0000, 0.0000, ],
# [-0.0000, 0.0000, 0.0417, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0000, 0.0000, 0.0000, ],
# [-0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0019, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ],])
    plot_the_poles_of(A, "Nominal System") # Visually check that the blimp is stable for the continuous case

    # Check that the blimp is controllable.
    C = np.block([np.linalg.matrix_power(A, i)@B for i in range(12)]) # R12 x 24
    if np.linalg.matrix_rank(C) == 12:
        print(f'The system is controllable by the PBH test')
    
    print(np.linalg.det(A))
    A_dis = scipy.linalg.expm(A)#np.eye(12) + dT * A
    B_int = np.zeros((12,12))
    for i in range(10000):
        dTau = dT / 10000
        tau = i * dTau
        B_int += scipy.linalg.expm(A * tau) * dTau
    B_dis = B_int @ B#np.linalg.inv(A) @ (A_dis - np.eye(12)) @ B
    print(B_dis)
    print(B * dT)
    
    # Show that without a feedback controller, the blimp is unstable when discretized.
    plot_the_poles_of(A_dis, "Nominal Discretized System", unit_circle=True)

    # Bring the poles in for the continuous time system.
    p = w
    for j in range(w.size):
        if np.imag(p[j]) > 1:
            p[j] -= 1 + 2.5j
        elif np.imag(p[j]) < -1:
            p[j] += -1 + 2.5j
    K = place_poles(A, B, p)

    A_fb = A - B@K.gain_matrix
    plot_the_poles_of(A_fb, "System with Feedback")

    A_fb_dis = scipy.linalg.expm(A_fb)
    plot_the_poles_of(A_fb_dis, "Discretized System with Feedback", unit_circle=True)
    
    print("Gain matrix:")
    print(np.round(K.gain_matrix, 5))

    w2,v2 = np.linalg.eig(A_fb)
    print(w2)
    A_dis = np.eye(12) + dT * A_fb
    w_dis,v_dis = np.linalg.eig(A_dis)
    print(w_dis)
    print(np.abs(w_dis))
    if np.max(np.abs(w_dis)) > 1:
        print('unstable.')
    # 2. Controllability. Is there a solution to R()


main()
