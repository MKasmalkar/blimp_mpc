from LinearBlimpSim import LinearBlimpSim
from NonlinearBlimpSim import NonlinearBlimpSim
from DiscreteBlimpSim import DiscreteBlimpSim

from OriginLQRController import OriginLQRController
from FeedbackLinearizedCtrlHelix import FeedbackLinearizedCtrlHelix

from BlimpPlotter import BlimpPlotter

import numpy as np

## PARAMETERS

dT = 0.05
STOP_TIME = 100

## SIMULATION

sim = NonlinearBlimpSim(dT)

plotter = BlimpPlotter()
plotter.init_plot('Linear')

ctrl = FeedbackLinearizedCtrlHelix(dT)
ctrl.init_sim(sim)

try:
    for n in range(int(STOP_TIME / dT)):
        u = ctrl.get_ctrl_action(sim)
        sim.update_model(u)
        plotter.update_plot(sim, ctrl)

        if plotter.window_was_closed():
            break
    
    if not plotter.window_was_closed():
        plotter.block()

except KeyboardInterrupt:
    print("Done!")

