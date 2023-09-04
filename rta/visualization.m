%% Begin main loop
load blimp-rta-processed.mat
% x = x';
t = t - t(1);
%%
t_end = max(t);

% From main_6dof.m
% t is a TIME x 1 double
% u is a 4 x TIME double
% x is as 12 x TIME double.

% Modify u so that it is the same time length as x.
u_d = u;%';
u = zeros(4, size(x, 2));
offset = size(x, 2) - size(u_d,2);
for i = 1:size(u_d, 2)
    u(:, i + offset) = u_d(:, i);
end
% t = linspace(0, t_end, size(x,2));
% Extract position/euler angle data from the state.
p = x(7:12, :);

%% Plot the results.
f = figure;
f.Position = [100 50 1200 700];

n_rows = 4;
n_cols = 3;
hold on;

subplot(n_rows, n_cols, 1);
plot(t, x(1, :), 'r');
legend('v_x');
xlabel('t (s)');
ylabel('m/s');
grid on;

subplot(n_rows, n_cols, 2);
plot(t, x(2, :), 'g');
legend('v_y');
xlabel('t (s)');
ylabel('m/s');
grid on;

subplot(n_rows, n_cols, 3);
plot(t, x(3, :), 'b');
legend('v_z');
xlabel('t (s)');
ylabel('m/s');
grid on;

subplot(n_rows, n_cols, 4);
plot(t, x(4, :), 'r');
legend('\omega_x');
xlabel('t (s)');
ylabel('rad / s');
grid on;

subplot(n_rows, n_cols, 5);
plot(t, x(5, :), 'g');
legend('\omega_y');
xlabel('t (s)');
ylabel('rad / s');
grid on;

subplot(n_rows, n_cols, 6);
plot(t, x(6, :), 'b');
legend('\omega_z');
xlabel('t (s)');
ylabel('rad / s');
grid on;

subplot(n_rows, n_cols, 7);
plot(t, x(7, :), 'r');
legend('x');
xlabel('t');
ylabel('m');
grid on;

subplot(n_rows, n_cols, 8);
plot(t, x(8, :), 'g');
legend('y');
xlabel('t');
ylabel('m');
grid on;

subplot(n_rows, n_cols, 9);
plot(t, x(9, :), 'b');
legend('z');
xlabel('t');
ylabel('m');
grid on;

subplot(n_rows, n_cols, 10);
plot(t, x(10, :), 'r');
legend('pitch \phi');
xlabel('t');
ylabel('rad');
grid on;

subplot(n_rows, n_cols, 11);
plot(t, x(11, :), 'g');
legend('roll \theta');
xlabel('t');
ylabel('rad');
grid on;

subplot(n_rows, n_cols, 12);
plot(t, x(12, :), 'b');
legend('yaw \psi');
xlabel('t');
ylabel('rad');
grid on;
