from LinearBlimpSim import LinearBlimpSim
from NonlinearBlimpSim import NonlinearBlimpSim
from DiscreteBlimpSim import DiscreteBlimpSim
from BlimpPlotter import BlimpPlotter
import numpy as np

dT = 0.05
STOP_TIME = 1

sim = LinearBlimpSim(dT)
plotter = BlimpPlotter()
plotter.init_plot('Linear')

# blimp_ctrl = BlimpController()

try:
    for i in range(int(STOP_TIME / dT)):
        u = np.array([0.05, 0, 0, 0])
        sim.update_model(u)
        plotter.update_plot(sim)

    plotter.block()

except KeyboardInterrupt:
    print("Done!")

