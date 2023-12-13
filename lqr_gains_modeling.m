

A_lin = [   0  cos(phi)  -1*w_y__b*sin(phi)  0  0  0  0 ;
            -0.154*cos(theta)  0.00979*v_z__b-0.0168  0  0  0.495*v_z__b+3.9e-4  0  0.495*v_x__b+0.00979*w_y__b ;
            -(w_y__b*sin(phi))/(sin(theta)^2-1)  sin(phi)*tan(theta)  w_y__b*cos(phi)*tan(theta)  1  0  0  0 ;
            0.154*sin(phi)*sin(theta)  0  -0.154*cos(phi)*cos(theta)  0.00979*v_z__b-0.0168  0  -0.495*v_z__b-3.9e-4  0.00979*w_x__b-0.495*v_y__b ;
            0  -1.62*v_z__b  0  0 -0.0249  0  -1.62*w_y__b ;
            0  0  0  1.62*v_z__b  0  -0.0249  1.62*w_x__b ;
            0  0.615*v_x__b+0.0244*w_y__b  0  0.0244*w_x__b-0.615*v_y__b  0.615*w_y__b  -0.615*w_x__b  -0.064 ;
        ];

B_lin = [    0, 0, 0
             0.0398, 0, 0
             0, 0, 0
             0, -0.0398, 0
             2.17, 0, 0
             0, 2.17, 0
             0, 0, 1.33
        ];

max_allowable_theta = 0.05;
max_allowable_phi = 0.05;

max_allowable_wy = 0.02;
max_allowable_wx = 0.02;

max_allowable_vx = 0.5;
max_allowable_vy = 0.5;

max_allowable_vz = 0.5;

Q = [1/max_allowable_theta^2 0 0 0 0 0 0
    0 1/max_allowable_wy^2 0 0 0 0 0
    0 0 1/max_allowable_phi^2 0 0 0 0
    0 0 0 1/max_allowable_wx^2 0 0 0
    0 0 0 0 1/max_allowable_vx^2 0 0
    0 0 0 0 0 1/max_allowable_vy^2 0
    0 0 0 0 0 0 1/max_allowable_vz^2];

R = eye(3);

