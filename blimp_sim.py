from re import L
from LinearBlimpSim import LinearBlimpSim
from NonlinearBlimpSim import NonlinearBlimpSim
from DiscreteBlimpSim import DiscreteBlimpSim

from OriginLQRController import OriginLQRController
from FeedbackLinearizedCtrlHelix import FeedbackLinearizedCtrlHelix
from WaypointTrackingMPC import WaypointTrackingMPC
from TestController import TestController

from BlimpPlotter import BlimpPlotter

import numpy as np

## PARAMETERS

dT = 0.05
STOP_TIME = 1

WINDOW_TITLE = 'Linear'

Simulator = LinearBlimpSim
Controller = TestController

## SIMULATION

sim = Simulator(dT)

plotter = BlimpPlotter()
plotter.init_plot(WINDOW_TITLE)

ctrl = Controller(dT)
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

