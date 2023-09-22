clear
close all
clc

computer_milp_data = readmatrix('outputs/time_log_computer.csv');
rpi_milp_data = readmatrix('outputs/time_log_rpi.csv');

disp("Computer, min: " + min(computer_milp_data))
disp("Computer, mean: " + mean(computer_milp_data))
disp("Computer, max: " + max(computer_milp_data))
disp(" ")
disp("RPi, min: " + min(rpi_milp_data))
disp("RPi, mean: " + mean(rpi_milp_data))
disp("RPi, max: " + max(rpi_milp_data))

computer_mpc_data = readmatrix('gur_veryfast_cmp.csv');
rpi_mpc_data = readmatrix('gur_veryfast_rpi.csv');

subplot(2, 1, 1)
plot(computer_milp_data*1e3)
hold on
plot(rpi_milp_data*1e3)
xlabel('Time step')
ylabel("Solve time (ms)")
title("Gurobi MILP Solve Time")
legend('Computer', 'RPi')

subplot(2, 1, 2)
plot(computer_mpc_data(:, 13)*1e3)
hold on
plot(rpi_mpc_data(:, 13)*1e3)
xlabel('Time step')
ylabel("Solve time (ms)")
title("Gurobi Simple MPC Solve Time")
legend("Computer", "RPi")