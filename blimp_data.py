from BlimpSim import BlimpSim
from LinearBlimpSim import LinearBlimpSim
from NonlinearBlimpSim import NonlinearBlimpSim
from DiscreteBlimpSim import DiscreteBlimpSim

from BlimpController import BlimpController
from OriginLQRController import OriginLQRController
from FeedbackLinearizedCtrlHelix import FeedbackLinearizedCtrlHelix
from WaypointTrackingMPC import WaypointTrackingMPC
from TestController import TestController
from MPCHelix import MPCHelix

from BlimpPlotter import BlimpPlotter
from BlimpLogger import BlimpLogger

import numpy as np
import time
import sys

## Parameters

TITLE = "Plots"

Simulator = NonlinearBlimpSim
Controller = FeedbackLinearizedCtrlHelix

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
plotter.init_plot(TITLE, True)

plotter.update_plot(sim, ctrl)
plotter.block()