from NonlinearBlimpSim import NonlinearBlimpSim

from FeedbackLinearizedCtrlHelix import FeedbackLinearizedCtrlHelix

from BlimpPlotter import BlimpPlotter
from BlimpLogger import BlimpLogger

import numpy as np
import sys

if len(sys.argv) < 2:
    print("Please run with output file name as argument.")
    sys.exit(0)

## PARAMETERS

dT = 0.05
STOP_TIME = 120
PLOT_WAVEFORMS = False

WINDOW_TITLE = 'Nonlinear'

Simulator = NonlinearBlimpSim
Controller = FeedbackLinearizedCtrlHelix

## SIMULATION

sim = Simulator(dT)

plotter = BlimpPlotter()
plotter.init_plot(WINDOW_TITLE, waveforms=PLOT_WAVEFORMS)

ctrl = Controller(dT)
ctrl.init_sim(sim)

if len(sys.argv) > 2:
    sim.load_data(sys.argv[2])
    ctrl.load_data(sys.argv[2])

try:
    for n in range(int(STOP_TIME / dT)):
        print("Time: " + str(round(n*dT, 2)))

        u = ctrl.get_ctrl_action(sim)
        sim.update_model(u)
        plotter.update_plot(sim, ctrl)
        
        if plotter.window_was_closed():
            break

except KeyboardInterrupt:
    print("Done!")
    sys.exit(0)

finally:
    logger = BlimpLogger(sys.argv[1])
    logger.log(sim, ctrl)

    if not plotter.window_was_closed():
        plotter.block()
