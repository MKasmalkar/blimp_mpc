close all
clc

%% Read data

if ~exist('data', 'var') || ~exist('data_short', 'var')
    data = readmatrix('benchmarks_matlab.xlsx');
    
    time = data(:, 1);
    
    solve_times = data(:, 2:7);
    position_error = data(:, 8:13);
    control_effort = data(:, 14:19);
    attitude_keeping = data(:, 20:25);
    
    % st = solve time
    % pe = position error
    % ce = control effort
    % ak = attitude-keeping
    
    % cmp = computer
    % rpi = raspberry pi
    
    % trk = tracking controller
    % gur = gurobi MPC
    % cas = casadi MPC
    
    st_cmp_trk = solve_times(:, 1);
    st_cmp_gur = solve_times(:, 2);
    st_cmp_cas = solve_times(:, 3);
    st_rpi_trk = solve_times(:, 4);
    st_rpi_gur = solve_times(:, 5);
    st_rpi_cas = solve_times(:, 6);
    
    pe_cmp_trk = position_error(:, 1);
    pe_cmp_gur = position_error(:, 2);
    pe_cmp_cas = position_error(:, 3);
    pe_rpi_trk = position_error(:, 4);
    pe_rpi_gur = position_error(:, 5);
    pe_rpi_cas = position_error(:, 6);
    
    ce_cmp_trk = control_effort(:, 1);
    ce_cmp_gur = control_effort(:, 2);
    ce_cmp_cas = control_effort(:, 3);
    ce_rpi_trk = control_effort(:, 4);
    ce_rpi_gur = control_effort(:, 5);
    ce_rpi_cas = control_effort(:, 6);
    
    ak_cmp_trk = attitude_keeping(:, 1);
    ak_cmp_gur = attitude_keeping(:, 2);
    ak_cmp_cas = attitude_keeping(:, 3);
    ak_rpi_trk = attitude_keeping(:, 4);
    ak_rpi_gur = attitude_keeping(:, 5);
    ak_rpi_cas = attitude_keeping(:, 6);

    data_short = readmatrix('benchmarks_short.xlsx');

    short_time = data_short(:, 1);
    short_st_rpi_gur = data_short(:, 2);
    short_st_rpi_cas = data_short(:, 3);
    short_pe_rpi_gur = data_short(:, 4);
    short_pe_rpi_cas = data_short(:, 5);
    short_ce_rpi_gur = data_short(:, 6);
    short_ce_rpi_cas = data_short(:, 7);
    short_ak_rpi_gur = data_short(:, 8);
    short_ak_rpi_cas = data_short(:, 9);
end

%% Plot

% Solve times

figure

subplot(211)
hold on
plot(time(5:end), st_cmp_trk(5:end)/1e6)
plot(time(5:end), st_cmp_gur(5:end)/1e6)
plot(time(5:end), st_cmp_cas(5:end)/1e6)

xlabel('Time (sec)')
ylabel('Solve time (ms)')
title('Solve times (computer)')

legend('Computer, Tracking', ...
       'Computer, Gurobi MPC', ...
       'Computer, Casadi MPC')

subplot(212)
hold on
plot(time(5:end), st_rpi_trk(5:end)/1e6)
plot(time(5:end), st_rpi_gur(5:end)/1e6)
plot(time(5:end), st_rpi_cas(5:end)/1e6)

xlabel('Time (sec)')
ylabel('Solve time (ms)')
title('Solve times (RPi)')

legend('RPi, Tracking', ...
       'RPi, Gurobi MPC', ...
       'RPi, Casadi MPC')


% Position error

figure
subplot(211)
hold on
plot(time, pe_cmp_trk)
plot(time, pe_cmp_gur)
plot(time, pe_cmp_cas)

xlabel('Time (sec)')
ylabel('Position error (m)')
title('Position error (computer)')

legend('Computer, Tracking', ...
       'Computer, Gurobi MPC', ...
       'Computer, Casadi MPC')

subplot(212)
hold on
plot(time, pe_rpi_trk)
plot(time, pe_rpi_gur)
plot(time, pe_rpi_cas)

xlabel('Time (sec)')
ylabel('Position error (m)')
title('Position error (RPi)')

legend('RPi, Tracking', ...
       'RPi, Gurobi MPC', ...
       'RPi, Casadi MPC')

% Control effort

figure
subplot(211)
hold on
plot(time, ce_cmp_trk)
plot(time, ce_rpi_trk)

xlabel('Time (sec)')
ylabel('Control effort (N)')
title('Control effort (tracking controller)')

legend('Computer', 'RPi')

subplot(212)
hold on
plot(time, ce_cmp_gur)
plot(time, ce_cmp_cas)
plot(time, ce_rpi_gur)
plot(time, ce_rpi_cas)

xlabel('Time (sec)')
ylabel('Control effort (N)')
title('Control effort (MPC)')

legend('Computer (Gurobi)', ...
        'Computer (Casadi)', ...
        'RPi (Gurobi)', ...
        'RPi (Casadi)')


% Attitude-keeping

figure
subplot(211)
hold on
plot(time, ak_cmp_trk*180/pi)
plot(time, ak_cmp_gur*180/pi)
plot(time, ak_cmp_cas*180/pi)

xlabel('Time (sec)')
ylabel('Norm of phi/theta error (deg)')
title('Attitude error (computer)')

legend('Computer, Tracking', ...
       'Computer, Gurobi MPC', ...
       'Computer, Casadi MPC')

subplot(212)
hold on
plot(time, ak_rpi_trk*180/pi)
plot(time, ak_rpi_gur*180/pi)
plot(time, ak_rpi_cas*180/pi)

xlabel('Time (sec)')
ylabel('Norm of phi/theta error (deg)')
title('Attitude error (RPi)')

legend('RPi, Tracking', ...
       'RPi, Gurobi MPC', ...
       'RPi, Casadi MPC')

% Short time horizon

figure

subplot(411)
hold on

plot(short_time, short_st_rpi_gur/1e6);
plot(short_time, short_st_rpi_cas/1e6);

xlabel('Time (sec)')

legend('Gurobi', 'Casadi')
title('Solve Time (ms)')

subplot(412)
hold on

plot(short_time, short_pe_rpi_gur);
plot(short_time, short_pe_rpi_cas);

xlabel('Time (sec)')

legend('Gurobi', 'Casadi')
title('Position error (m)')

subplot(413)
hold on

plot(short_time, short_ce_rpi_gur);
plot(short_time, short_ce_rpi_cas);

xlabel('Time (sec)')

legend('Gurobi', 'Casadi')
title('Control effort (N)')

subplot(414)
hold on

plot(short_time, short_ak_rpi_gur*180/pi);
plot(short_time, short_ak_rpi_cas*180/pi);

xlabel('Time (sec)')

legend('Gurobi', 'Casadi')
title('Norm of attitude error (deg)')