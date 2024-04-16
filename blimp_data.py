from NonlinearBlimpSim import NonlinearBlimpSim

from CBF import CBF

from BlimpPlotter import BlimpPlotter
from BlimpLogger import BlimpLogger

import numpy as np
import time
import sys

## Parameters

TITLE = "Plots"

# Neither of these selections matter - these objects
# just need to be created in order to load and plot
# the simulation data from the file.

Simulator = NonlinearBlimpSim
Controller = CBF

PLOT_ANYTHING = True
PLOT_WAVEFORMS = False

## Plotting

if len(sys.argv) < 2:
    print("Please run with data file name as first argument.")
    sys.exit(0)

dT = 0.05  # will be overridden by data load anyways
sim = Simulator(dT)
ctrl = Controller(dT)
plotter = BlimpPlotter()

sim.load_data(sys.argv[1])
ctrl.load_data(sys.argv[1])

plotter.init_plot(TITLE,
                  waveforms=PLOT_WAVEFORMS,
                  disable_plotting=(not PLOT_ANYTHING))

plotter.update_plot(sim, ctrl)
plotter.block()