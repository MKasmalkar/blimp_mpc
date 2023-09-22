clear
close all
clc

disp("Casadi:")
analyze_performance('Casadi', ...
                    'run1.csv', ...
                    'rpi_run2.csv');
disp(' ')
disp("Gurobi:")
analyze_performance('Gurobi', ...
                    'gur_veryfast_cmp.csv', ...
                    'gur_veryfast_rpi.csv');

analyze_performance('Nonlinear', ...
                    'nonlinear_1.csv', ...
                    'nonlinear_1.csv');

function analyze_performance(name, computer_csv, rpi_csv)
    data_computer = readmatrix(computer_csv);
    
    time_comp = data_computer(:, 1);
    X_comp = data_computer(:, 2);
    Y_comp = data_computer(:, 3);
    Z_comp = data_computer(:, 4);
    refX_comp = data_computer(:, 5);
    refY_comp = data_computer(:, 6);
    refZ_comp = data_computer(:, 7);
    error_comp = data_computer(:, 8);
    u0_comp = data_computer(:, 9);
    u1_comp = data_computer(:, 10);
    u2_comp = data_computer(:, 11);
    u3_comp = data_computer(:, 12);
    deltaT_comp = data_computer(:, 13);
    
    data_rpi = readmatrix(rpi_csv);
    
    time_rpi = data_rpi(:, 1);
    X_rpi = data_rpi(:, 2);
    Y_rpi = data_rpi(:, 3);
    Z_rpi = data_rpi(:, 4);
    refX_rpi = data_rpi(:, 5);
    refY_rpi = data_rpi(:, 6);
    refZ_rpi = data_rpi(:, 7);
    error_rpi = data_rpi(:, 8);
    u0_rpi = data_rpi(:, 9);
    u1_rpi = data_rpi(:, 10);
    u2_rpi = data_rpi(:, 11);
    u3_rpi = data_rpi(:, 12);
    deltaT_rpi = data_rpi(:, 13);
    
    
    figure("Name", name)
    subplot(3, 3, 1:6)
    plot3(X_rpi, Y_rpi, Z_rpi, 'Color', 'b');
    xlabel('x')
    ylabel('y')
    zlabel('z')
    hold on
    scatter3(refX_comp, refY_comp, refZ_comp, 50, 'r', 'filled')
    title("Blimp path")
    subtitle("Red markers indicate waypoints")
    
    subplot(3, 3, 7:9)
    plot(time_comp, deltaT_comp)
    hold on
    plot(time_rpi, deltaT_rpi)
    xlabel('Simulation time (sec)')
    ylabel('Computation time (sec)')
    yticks(0:0.1:1)
    ylim([0 1])
    legend('Computer', 'RPi')
    title('Optimization Program Solve Time')
    
    disp("Average solve time on computer: " + mean(deltaT_comp))
    disp("Peak solve time on computer (exluding initial startup): " + max(deltaT_comp(2:end)))
    disp("Min solve time on computer: " + min(deltaT_comp))
    disp(' ')
    disp("Average solve time on RPi: " + mean(deltaT_rpi))
    disp("Peak solve time on RPi (excluding initial startup): " + max(deltaT_rpi(2:end)))
    disp("Min solve time on RPi: " + min(deltaT_rpi))
end