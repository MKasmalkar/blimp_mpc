clc
close all

if ~exist('kmatrix', 'var')
    kmatrix = readmatrix('kmatrix.xlsx');
end

time = kmatrix(:, 1);

theta = kmatrix(:, 2);
w_y = kmatrix(:, 3);
phi = kmatrix(:, 4);
w_x = kmatrix(:, 5);
v_x = kmatrix(:, 6);
v_y = kmatrix(:, 7);
v_z = kmatrix(:, 8);

statevars = [theta, w_y, phi, w_x, v_x, v_y, v_z];

% rows, i.e. k1[1:7], etc.
k1 = kmatrix(:, 9:15);
k2 = kmatrix(:, 16:22);
k3 = kmatrix(:, 23:29);

k11 = k1(:, 1);
k12 = k1(:, 2);
k13 = k1(:, 3);
k14 = k1(:, 4);
k15 = k1(:, 5);
k16 = k1(:, 6);
k17 = k1(:, 7);

k21 = k2(:, 1);
k22 = k2(:, 2);
k23 = k2(:, 3);
k24 = k2(:, 4);
k25 = k2(:, 5);
k26 = k2(:, 6);
k27 = k2(:, 7);

k31 = k3(:, 1);
k32 = k3(:, 2);
k33 = k3(:, 3);
k34 = k3(:, 4);
k35 = k3(:, 5);
k36 = k3(:, 6);
k37 = k3(:, 7);

%% State variables
% theta, w_y, phi, w_x, v_x, v_y, v_z

figure
subplot(711)
plot(time, theta)
title('theta')
hold on
subplot(712)
plot(time, w_y)
title('w_y')
subplot(713)
plot(time, phi)
title('phi')
subplot(714)
plot(time, w_x)
title('w_x')
subplot(715)
plot(time, v_x)
title('v_x')
subplot(716)
plot(time, v_y)
title('v_y')
subplot(717)
plot(time, v_z)
title('v_z')
xlabel('Time')

figure
subplot(711)
plot(time, k11)
title('Fx, theta')
hold on
subplot(712)
plot(time, k12)
title('Fx, w_y')
subplot(713)
plot(time, k13)
title('Fx, phi')
subplot(714)
plot(time, k14)
title('Fx, w_x')
subplot(715)
plot(time, k15)
title('Fx, v_x')
subplot(716)
plot(time, k16)
title('Fx, v_y')
subplot(717)
plot(time, k17)
title('Fx, v_z')
xlabel('Time')

figure
subplot(711)
plot(time, k21)
title('Fy, theta')
hold on
subplot(712)
plot(time, k22)
title('Fy, w_y')
subplot(713)
plot(time, k23)
title('Fy, phi')
subplot(714)
plot(time, k24)
title('Fy, w_x')
subplot(715)
plot(time, k25)
title('Fy, v_x')
subplot(716)
plot(time, k26)
title('Fy, v_y')
subplot(717)
plot(time, k27)
title('Fy, v_z')
xlabel('Time')

figure
subplot(711)
plot(time, k31)
title('Fz, theta')
hold on
subplot(712)
plot(time, k32)
title('Fz, w_y')
subplot(713)
plot(time, k33)
title('Fz, phi')
subplot(714)
plot(time, k34)
title('Fz, w_x')
subplot(715)
plot(time, k35)
title('Fz, v_x')
subplot(716)
plot(time, k36)
title('Fz, v_y')
subplot(717)
plot(time, k37)
title('Fz, v_z')
xlabel('Time')

statevar_names = ["the", "w_y", "phi", "w_x", "v_x", "v_y", "v_z"];
gains_k_names = ["K11", "K12", "K13", "K14", "K15", "K16", "K17", ...
               "K21", "K22", "K23", "K24", "K25", "K26", "K27", ...
               "K31", "K32", "K33", "K34", "K35", "K36", "K37"];
gains_names = ["fx, the", "fx, wy", "fx, phi", "fx, wx", "fx, vx", "fx, vy", "fx, vz", ...
               "fy, the", "fy, wy", "fy, phi", "fy, wx", "fy, vx", "fy, vy", "fy, vz", ...
               "fz, the", "fz, wy", "fz, phi", "fz, wx", "fz, vx", "fz, vy", "fz, vz"];

the_correlations = zeros(1, 21);
w_y_correlations = zeros(1, 21);
phi_correlations = zeros(1, 21);
w_x_correlations = zeros(1, 21);
v_x_correlations = zeros(1, 21);
v_y_correlations = zeros(1, 21);
v_z_correlations = zeros(1, 21);

the_correlations(1)  = get_corr(theta, k11);
the_correlations(2)  = get_corr(theta, k12);
the_correlations(3)  = get_corr(theta, k13);
the_correlations(4)  = get_corr(theta, k14);
the_correlations(5)  = get_corr(theta, k15);
the_correlations(6)  = get_corr(theta, k16);
the_correlations(7)  = get_corr(theta, k17);
the_correlations(8)  = get_corr(theta, k21);
the_correlations(9)  = get_corr(theta, k22);
the_correlations(10) = get_corr(theta, k23);
the_correlations(11) = get_corr(theta, k24);
the_correlations(12) = get_corr(theta, k25);
the_correlations(13) = get_corr(theta, k26);
the_correlations(14) = get_corr(theta, k27);
the_correlations(15) = get_corr(theta, k31);
the_correlations(16) = get_corr(theta, k32);
the_correlations(17) = get_corr(theta, k33);
the_correlations(18) = get_corr(theta, k34);
the_correlations(19) = get_corr(theta, k35);
the_correlations(20) = get_corr(theta, k36);
the_correlations(21) = get_corr(theta, k37);

w_y_correlations(1)  = get_corr(w_y, k11);
w_y_correlations(2)  = get_corr(w_y, k12);
w_y_correlations(3)  = get_corr(w_y, k13);
w_y_correlations(4)  = get_corr(w_y, k14);
w_y_correlations(5)  = get_corr(w_y, k15);
w_y_correlations(6)  = get_corr(w_y, k16);
w_y_correlations(7)  = get_corr(w_y, k17);
w_y_correlations(8)  = get_corr(w_y, k21);
w_y_correlations(9)  = get_corr(w_y, k22);
w_y_correlations(10) = get_corr(w_y, k23);
w_y_correlations(11) = get_corr(w_y, k24);
w_y_correlations(12) = get_corr(w_y, k25);
w_y_correlations(13) = get_corr(w_y, k26);
w_y_correlations(14) = get_corr(w_y, k27);
w_y_correlations(15) = get_corr(w_y, k31);
w_y_correlations(16) = get_corr(w_y, k32);
w_y_correlations(17) = get_corr(w_y, k33);
w_y_correlations(18) = get_corr(w_y, k34);
w_y_correlations(19) = get_corr(w_y, k35);
w_y_correlations(20) = get_corr(w_y, k36);
w_y_correlations(21) = get_corr(w_y, k37);

phi_correlations(1)  = get_corr(phi, k11);
phi_correlations(2)  = get_corr(phi, k12);
phi_correlations(3)  = get_corr(phi, k13);
phi_correlations(4)  = get_corr(phi, k14);
phi_correlations(5)  = get_corr(phi, k15);
phi_correlations(6)  = get_corr(phi, k16);
phi_correlations(7)  = get_corr(phi, k17);
phi_correlations(8)  = get_corr(phi, k21);
phi_correlations(9)  = get_corr(phi, k22);
phi_correlations(10) = get_corr(phi, k23);
phi_correlations(11) = get_corr(phi, k24);
phi_correlations(12) = get_corr(phi, k25);
phi_correlations(13) = get_corr(phi, k26);
phi_correlations(14) = get_corr(phi, k27);
phi_correlations(15) = get_corr(phi, k31);
phi_correlations(16) = get_corr(phi, k32);
phi_correlations(17) = get_corr(phi, k33);
phi_correlations(18) = get_corr(phi, k34);
phi_correlations(19) = get_corr(phi, k35);
phi_correlations(20) = get_corr(phi, k36);
phi_correlations(21) = get_corr(phi, k37);

w_x_correlations(1)  = get_corr(w_x, k11);
w_x_correlations(2)  = get_corr(w_x, k12);
w_x_correlations(3)  = get_corr(w_x, k13);
w_x_correlations(4)  = get_corr(w_x, k14);
w_x_correlations(5)  = get_corr(w_x, k15);
w_x_correlations(6)  = get_corr(w_x, k16);
w_x_correlations(7)  = get_corr(w_x, k17);
w_x_correlations(8)  = get_corr(w_x, k21);
w_x_correlations(9)  = get_corr(w_x, k22);
w_x_correlations(10) = get_corr(w_x, k23);
w_x_correlations(11) = get_corr(w_x, k24);
w_x_correlations(12) = get_corr(w_x, k25);
w_x_correlations(13) = get_corr(w_x, k26);
w_x_correlations(14) = get_corr(w_x, k27);
w_x_correlations(15) = get_corr(w_x, k31);
w_x_correlations(16) = get_corr(w_x, k32);
w_x_correlations(17) = get_corr(w_x, k33);
w_x_correlations(18) = get_corr(w_x, k34);
w_x_correlations(19) = get_corr(w_x, k35);
w_x_correlations(20) = get_corr(w_x, k36);
w_x_correlations(21) = get_corr(w_x, k37);

v_x_correlations(1)  = get_corr(v_x, k11);
v_x_correlations(2)  = get_corr(v_x, k12);
v_x_correlations(3)  = get_corr(v_x, k13);
v_x_correlations(4)  = get_corr(v_x, k14);
v_x_correlations(5)  = get_corr(v_x, k15);
v_x_correlations(6)  = get_corr(v_x, k16);
v_x_correlations(7)  = get_corr(v_x, k17);
v_x_correlations(8)  = get_corr(v_x, k21);
v_x_correlations(9)  = get_corr(v_x, k22);
v_x_correlations(10) = get_corr(v_x, k23);
v_x_correlations(11) = get_corr(v_x, k24);
v_x_correlations(12) = get_corr(v_x, k25);
v_x_correlations(13) = get_corr(v_x, k26);
v_x_correlations(14) = get_corr(v_x, k27);
v_x_correlations(15) = get_corr(v_x, k31);
v_x_correlations(16) = get_corr(v_x, k32);
v_x_correlations(17) = get_corr(v_x, k33);
v_x_correlations(18) = get_corr(v_x, k34);
v_x_correlations(19) = get_corr(v_x, k35);
v_x_correlations(20) = get_corr(v_x, k36);
v_x_correlations(21) = get_corr(v_x, k37);

v_y_correlations(1)  = get_corr(v_y, k11);
v_y_correlations(2)  = get_corr(v_y, k12);
v_y_correlations(3)  = get_corr(v_y, k13);
v_y_correlations(4)  = get_corr(v_y, k14);
v_y_correlations(5)  = get_corr(v_y, k15);
v_y_correlations(6)  = get_corr(v_y, k16);
v_y_correlations(7)  = get_corr(v_y, k17);
v_y_correlations(8)  = get_corr(v_y, k21);
v_y_correlations(9)  = get_corr(v_y, k22);
v_y_correlations(10) = get_corr(v_y, k23);
v_y_correlations(11) = get_corr(v_y, k24);
v_y_correlations(12) = get_corr(v_y, k25);
v_y_correlations(13) = get_corr(v_y, k26);
v_y_correlations(14) = get_corr(v_y, k27);
v_y_correlations(15) = get_corr(v_y, k31);
v_y_correlations(16) = get_corr(v_y, k32);
v_y_correlations(17) = get_corr(v_y, k33);
v_y_correlations(18) = get_corr(v_y, k34);
v_y_correlations(19) = get_corr(v_y, k35);
v_y_correlations(20) = get_corr(v_y, k36);
v_y_correlations(21) = get_corr(v_y, k37);

v_z_correlations(1)  = get_corr(v_z, k11);
v_z_correlations(2)  = get_corr(v_z, k12);
v_z_correlations(3)  = get_corr(v_z, k13);
v_z_correlations(4)  = get_corr(v_z, k14);
v_z_correlations(5)  = get_corr(v_z, k15);
v_z_correlations(6)  = get_corr(v_z, k16);
v_z_correlations(7)  = get_corr(v_z, k17);
v_z_correlations(8)  = get_corr(v_z, k21);
v_z_correlations(9)  = get_corr(v_z, k22);
v_z_correlations(10) = get_corr(v_z, k23);
v_z_correlations(11) = get_corr(v_z, k24);
v_z_correlations(12) = get_corr(v_z, k25);
v_z_correlations(13) = get_corr(v_z, k26);
v_z_correlations(14) = get_corr(v_z, k27);
v_z_correlations(15) = get_corr(v_z, k31);
v_z_correlations(16) = get_corr(v_z, k32);
v_z_correlations(17) = get_corr(v_z, k33);
v_z_correlations(18) = get_corr(v_z, k34);
v_z_correlations(19) = get_corr(v_z, k35);
v_z_correlations(20) = get_corr(v_z, k36);
v_z_correlations(21) = get_corr(v_z, k37);

figure

subplot(241)
bar(the_correlations)
title("theta")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

subplot(242)
bar(w_y_correlations)
title("w_y")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

subplot(243)
bar(phi_correlations)
title("phi")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

subplot(244)
bar(w_x_correlations)
title("w_y")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

subplot(245)
bar(v_x_correlations)
title("v_x")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

subplot(246)
bar(v_y_correlations)
title("v_y")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

subplot(247)
bar(v_z_correlations)
title("v_z")
set(gca, 'XTickLabel', gains_names, 'XTick', 1:numel(gains_names), 'FontSize', 8);

function y = get_corr(a, b)
    c = corrcoef(a,b);
    y = c(2,1);
end