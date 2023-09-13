clear
close all
clc

data = readmatrix("run1.csv");

time = data(:, 1);
X = data(:, 2);
Y = data(:, 3);
Z = data(:, 4);
refX = data(:, 5);
refY = data(:, 6);
refZ = data(:, 7);
error = data(:, 8);
u0 = data(:, 9);
u1 = data(:, 10);
u2 = data(:, 11);
u3 = data(:, 12);
deltaT = data(:, 13);

figure
subplot(3, 3, 1:6)
plot3(X, Y, Z, 'Color', 'b');
xlabel('x')
ylabel('y')
zlabel('z')
hold on
scatter3(refX, refY, refZ, 50, 'r', 'filled')

subplot(3, 3, 7:9)
plot(time, deltaT)
xlabel('Simulation time (sec)')
ylabel('Computation time (sec)')