
from utilities import *
from LinearBlimpSim import LinearBlimpSim

sim = LinearBlimpSim(0.05)

matrix_to_matlab(sim.get_A_lin(), "A_lin")
print()
matrix_to_matlab(sim.get_B_lin(), "B_lin")
print()
matrix_to_matlab(sim.get_A_dis(), "A_dis")
print()
matrix_to_matlab(sim.get_B_dis(), "B_dis")