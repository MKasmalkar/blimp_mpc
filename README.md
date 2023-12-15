# Blimp MPC Code

## Entry points

The entry points into the code are as follows
* `blimp_sim.py`
* `blimp_data.py`

`blimp_sim.py` is used to run a simulation. Invoking `blimp_sim.py` is done as follows:

`python blimp_sim.py log_file.csv \[start_file.csv\]`

`log_file`.csv is the file to which data should be logged. Logs are stored in `logs/\[log_file.csv\]`. I often copy/pasted logs into the `logs_excel` folder to do Excel/MATLAB analysis of them without touching the raw data.
`start_file.csv` is an optional argument specifying the file name of a previously logged set of data. The simulation will start from the last recorded state/timestep in this file.

Once the simulation is running, it can be stopped with Ctrl-C or by closing the display window if plotting is enabled. The data up to the current point will be logged to the log file.

`blimp_data.py` is used to view simulation results from a logged file. It is invoked as follows:

`python blimp_data.py log_file.csv`

`log_file.csv` is the file containing the logged data to display.

## Configuring the simulation

Within `blimp_sim.py`, the simulation settings can be configured in the `## PARAMETERS` section at the top of the file.

* The simulated time step is set with the variable `dT`
* The time to simulate for is set with `STOP_TIME`
* `PLOT_ANYTHING` determines whether the trajectory will be plotted. If this flag is set and `PLOT_WAVEFORM` is not set, then the only thing that will be plotted is the 3D trajectory and the blimp's progress along this trajectory.
* `PLOT_WAVEFORMS` determines whether rectangular plots of the data will be shown alongside the blimp's trajectory. If this flag is set and `PLOT_ANYTHING` is also set, then both the 3D trajectory and the rectangular plots of waveforms will be displayed. If this flag is set but `PLOT_WAVEFORMS` is not set, nothing will be plotted.
* `WINDOW_TITLE` sets the title of the display window used to show the trajectory/waveforms.
* `Simulator` is used to select which simulator class is used for the simulation (see more below). This is used, for example, to select between a CT linear, CT nonlinear, or discrete-time simulation.
* `Controller` is used to select the controller class to be used for the simulation (see more below). This is used to select the desired control law; for instance, the tracking controller, an MPC, etc.

## Simulator classes

Simulator objects define the nature of the simulation dynamics. The superclass `BlimpSim` in `BlimpSim.py` defines a number of methods and parameters common to all simulators, but it also defines the `update_model(u)` method, which is the main function of each simulator. Subclasses of `BlimpSim` override the `update_model(u)` and implement their respective model dynamics there. It accepts a control input vector, `u`, which is the control input to be applied to the plant. Subclasses are responsible for then calling the `update_history()` method, which updates the time step and all logged data with the new model state.

The following simulator classes have been created:
* `NonlinearBlimpSim`
* `LinearBlimpSim`
* `DiscreteBlimpSim`

## Controller classes

Controller objects define the control law to be implemented. They derive from the `BlimpController` superclass in `BlimpController.py` and override the `get_ctrl_action(sim)` method, which implements the control law computation and updates the passed simulator object with the computed control law.

The following controller classes have been created (there are some other miscellaneous ones as well but these are the most recent/relevant ones):
* `CasadiHelix`: linear MPC for helix tracking using Casadi
* `CasadiNonlinearHelix`: nonlinear MPC for helix tracking using Casadi
* `MPCHelix`: linear MPC for helix tracking using Gurobi
* `MPCNonlinearHelix`: nonlinear MPC for helix tracking using Gurobi (doesn't work)
* `OriginLQRController`: regulates blimp to the origin using LQR
* `TrackingHelixTrajGen`: feedback-linearized controller using logistic shaping of time progression along helical trajectory
* `TrackingLineTrajGen`: feedback-linearized controller using logistic shaping of time progressiona along linear trajectory
* `TrackingLine`: feedback-linearized controller following a linear trajectory
* `TrackingNoDamping`: feedback-linearized controller following helical trajectory, no zero dynamics damping
* `TrackingOneshotDecoupledAttDyn`: feedback-linearized controller following helical trajectory; uses two independent, pre-computed LQR matrices for decoupled x and y zero dynamics
* `TrackingOneshotFullOrder`: feedback-linearized controller following helical trajectory; uses 12th order model for zero dynamics damping with pre-computed LQR gains
* `TrackingOneShotReducedOrder`: feedback-linearized controller following helical trajectory; uses 7th order model for zero dynamics damping with pre-computed LQR gains
* `TrackingRepeatedFullOrder`: feedback-linearized controller following helical trajectory; uses 12th order model for zero dynamics damping with online LQR gain computation
* `TrackingRepeatedReducedOrder`: feedback-linearized controller following helical trajectory; uses 7th order model for zero dynamics damping with online LQR gain computation
* `TrackingRepeatedReducedOrderLogging`: same as `TrackingRepeatedReducedOrder` except also logs K gains to a file
* `WaypointTrackingMPC`: Gurobi MPC for navigating towards a series of waypoints using linear dynamical simulation
* `WaypointTrackingNonlinearMPC`: poorly named, Gurobi MPC for navigating waypoints using nonlinear dynamical simulation; currently same as `WaypointTrackingMPC` but in principle the time horizon could be different

## Other important files

* `parameters.py`: this is where the mechanical parameters (dimensions, weights, coefficients, etc.) of the blimp are defined
* `operators.py`: this is where operators such as rotations or other computations are defined as functions

Luke's RTA code is in the `rta` folder. I made one edit to this code: in `blimp.py`, I added a negative sign to `self.r_b_z_tb` on line 67 to correctly account for the direction of the torque induced by y force.
