# Author: Luke Baird
# Purpose: perform STL post-processing on state data
import mtl
import numpy as np
import matplotlib.animation as animation
import scipy.io
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

mat = scipy.io.loadmat('blimp-rta-output.mat')
# mat = scipy.io.loadmat('blimp-rta-output-no-gyro-yes-tube.mat')
# print(mat['x_projected'].shape)
# x = mat['x_projected'][:, 0, ].T
x = mat['x']
x_projected = mat['x_projected']
t = mat['t'].squeeze(0)
# t = t[10:]#t[3:] # t[10:]
t -= t[0]

# cut off the last few values of x and t.
cutoff_index = x_projected.shape[0]
x = x[:, :cutoff_index]
t = t[:cutoff_index]

cleanup_index = 70 # Cut this many values.
x = x[:, cleanup_index:]
t = t[:-cleanup_index] # do the end for t.
# u = u[:, cleanup_index:] # Later, once we extract it.
x_projected = x_projected[cleanup_index:, :, :]

print(x.shape)
print(t.shape)
print(x_projected.shape)

# exit()

# STEP 2: CALCULATE DEVIATION FROM TO A SQUARE OF RADIUS 1.41
square_dim = 1.23 # 1.38 # remove the 0.03 buffer, and the radius (not the diameter) of the blimp.
num_time_steps = t.shape[0]
dist = -np.ones((num_time_steps,)) * 500 # something gigantic.
for i in range(num_time_steps):
    for j in [6, 7]: # positions
        if x[j, i] >= 0:
            dist[i] = max(x[j,i] - square_dim, dist[i])
        else:
            dist[i] = max(- (x[j,i] + square_dim), dist[i])
rho = np.zeros((num_time_steps,))

# STEP 3: CONSTRUCT THE STL FORMULA.
x_mtl = mtl.parse('x')
dT = 0.25

phi1 = x_mtl.always(lo=0, hi=round(1/dT) - 1) # I think this is the truth as to what I actually enforced.
phi2 = x_mtl.eventually(lo=0, hi=round(5/dT))#phi1.eventually(lo=0, hi=round(3/dT)) # phi1

phi = x_mtl | phi2

# STEP 3.1: HEIGHT SPECFICATION
h_mtl = mtl.parse('h') # We just require this, but we also need the height trace.
height_trace = np.zeros((num_time_steps,))
for i in range(num_time_steps): # work with x[8,:]
    height_trace[i] = min(2.5 - x[8,i], x[8,i] - 1.0) # x[8,i] + 1.0, if the bound is -1.
rho_height = np.zeros((num_time_steps,1))

# Construct the data.
dist_as_list_data = dist.tolist()
height_as_list_data = height_trace.tolist()
for k in range(num_time_steps):
    dist_as_list_data[k] = (k, dist_as_list_data[k])
    height_as_list_data[k] = (k, height_as_list_data[k])
for k in range(num_time_steps):
    data = {
        'x' : dist_as_list_data[k:k+50]#50] 
    }
    data_height = {
        'h': height_as_list_data#[k]
    }
    rho[k] = phi(data, time=k)
    rho_height[k, 0] = h_mtl(data_height, time=k)
lineTop = np.ones((num_time_steps,)) * square_dim
lineBottom = np.ones((num_time_steps,)) * square_dim * -1

x_fig, (x_axes, rho_axes) = plt.subplots(2,)
t_range = [t * dT for t in range(num_time_steps)]
x_axes.plot(t_range, dist, 'r.-')
rho_axes.plot(t_range, rho, 'b.-')
rho_axes.set_title('$\\rho$')
rho_axes.grid(True)
x_axes.grid(True)
x_axes.set_title('distance from the square (tube MPC)')
x_fig.savefig('output/robustness_post_processing.pdf', format='pdf')

# exit()

a_fig, (a_ax, b_ax) = plt.subplots(2,)
a_ax.plot(t_range, x[6,:], 'r.-')
a_ax.plot(t_range, lineTop, 'k')
a_ax.plot(t_range, lineBottom, 'k')
b_ax.plot(t_range, x[7,:], 'b.-')
b_ax.plot(t_range, lineTop, 'k')
b_ax.plot(t_range, lineBottom, 'k')
a_ax.set_title('$x(t)$')
b_ax.set_title('$y(t)$')
a_fig.savefig('output/xy_plot.pdf', format='pdf')

phi_fig, (p_ax, t_ax) = plt.subplots(2,)
p_ax.plot(t_range, x[9,:], 'r.-')
t_ax.plot(t_range, x[10,:], 'b.-')
p_ax.set_title('$\\theta(t)$')
t_ax.set_title('$\\phi(t)$')
phi_fig.savefig('output/angles_plot.pdf', format='pdf')

h_fig, h_ax = plt.subplots(1,) # h_rho_ax
h_ax.plot(t_range, x[8,:], 'r.-')
h_ax.plot(t_range, np.ones((num_time_steps,)) * -2, 'k')
# h_ax.plot(t_range, np.ones((num_time_steps,)) * 1.0, 'k')
h_ax.set_ylabel('altitude')
h_ax.set_xlabel('time (s)')
h_ax.grid(True)
# h_rho_ax.plot(t_range, rho_height, 'b.-')
# h_rho_ax.set_ylabel('$h \leq 2.5 \wedge h \geq 1.0$')
# h_rho_ax.set_xlabel('time (s)')
# h_rho_ax.grid(True)
# h_ax.set_title('height specification plots')
h_fig.savefig('output/height.pdf', format='pdf')


fig = plt.figure(figsize=(8, 4.5))
ax = fig.add_subplot(autoscale_on=False)
ax.set_xlim(0, num_time_steps / 4)
ax.set_ylim(-0.8, 1.4)
ax.set_xlabel('$t (s)$')
ax.set_ylabel('$\\rho[t]$')
ax.set_title('$\phi$ main formula robustness')
ax.grid()
robustness, = ax.plot([], [], 'b.-')
fig.savefig('output/robustness.pdf', format='pdf')


def animate(i):
    # print(f'calling animate {i}')
    this_past_t = t[:i]
    this_past_rho = rho[:i]
    robustness.set_data(this_past_t, this_past_rho)
    return [robustness]

ani = animation.FuncAnimation(
    fig, animate, num_time_steps, interval=int(100 * dT), blit=True, repeat=False
) # speed is 10x

writer = animation.PillowWriter(fps=int(10/dT))#animation.writers['ffmpeg'](fps=30)
ani.save('output/y.gif', writer=writer)

# Create an animation for the blimp flying around.
square_x = np.array([-square_dim, -square_dim, square_dim, square_dim, -square_dim])
square_y = np.array([-square_dim, square_dim, square_dim, -square_dim, -square_dim])

flying_figure = plt.figure(figsize=(7, 7))
flying_ax = flying_figure.add_subplot(autoscale_on=False)
flying_ax.set_xlim(-3, 3)
flying_ax.set_ylim(-3, 3)
flying_ax.set_xlabel('$x$')
flying_ax.set_ylabel('$y$')
flying_ax.set_title('2D plot of trajectory - 2x speed')
flying_ax.plot(square_x, square_y, 'r')
flying_ax.grid()
flying_trajectory, = flying_ax.plot([], [], 'k.-')
flying_projected, = flying_ax.plot([], [], 'g.-')
flying_ax.legend(['Square', 'Historical', 'Safe Backup'])
def flying_animation(i):
    past_x = x[6,:i]
    past_y = x[7,:i]
    projected_x = x_projected[i, 6, :]
    projected_y = x_projected[i, 7, :]
    flying_trajectory.set_data(past_x, past_y)
    flying_projected.set_data(projected_x, projected_y)
    return [flying_trajectory, flying_projected]
flying_animation = animation.FuncAnimation(
    flying_figure, flying_animation, num_time_steps, interval=int(1000*dT/2), blit=True, repeat=False # speed: 4x
)
flyingWriter = animation.PillowWriter(fps=int(2/dT))
flying_animation.save('output/flying.gif', writer=flyingWriter)

# TODO: for debugging purposes, it could be useful to plot the states x and y alongside the distance
plt.show()