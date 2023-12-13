import numpy as np
import casadi as cs

##  Operators
def H(r):
    return np.block([ [np.identity(3),         S(r).T],
                      [np.zeros((3, 3)),        np.identity(3)]])

def S(r):
    return np.block([[  0,       -r[2],    r[1]],
                     [r[2],     0,      -r[0]],
                     [-r[1],    r[0],      0]])

def C(M, nu):
    dimM = np.shape(M)[0]

    M11 = M[0:int(dimM/2), 0:int(dimM/2)]
    M12 = M[0:int(dimM/2), int(dimM/2):dimM]
    M21 = M[int(dimM/2):dimM, 0:int(dimM/2)]
    M22 = M[int(dimM/2):dimM, int(dimM/2):dimM]

    dimNu = np.shape(nu)[0]
    nu1 = nu[0:int(dimNu/2)]
    nu2 = nu[int(dimNu/2):dimNu]

    return np.block([[ np.zeros((3, 3)),    -S(M11@nu1 + M12@nu2)],
                     [-S(M11@nu1 + M12@nu2), -S(M21@nu1 + M22@nu2)]])


## Rotation matrices

def R_b__n(phi, theta, psi):

    x_rot = np.array([[1,         0,           0],
                      [0,       np.cos(phi),   -np.sin(phi)],
                      [0,       np.sin(phi),    np.cos(phi)]])

    y_rot = np.array([[np.cos(theta),      0,        np.sin(theta)],
                      [0,         1,           0],
                      [-np.sin(theta),     0,        np.cos(theta)]])
    
    z_rot = np.array([[np.cos(psi),    -np.sin(psi),       0],
                      [np.sin(psi),     np.cos(psi),       0],
                      [0,          0,           1]])

    # World-to-body
    return z_rot @ y_rot @ x_rot

def R_b__n_inv(phi, theta, psi):

    return np.array([[np.cos(psi)*np.cos(theta), np.cos(theta)*np.sin(psi), -np.sin(theta)],
                     [np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta), np.cos(theta)*np.sin(phi)],
                     [np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta), np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi), np.cos(phi)*np.cos(theta)]])

def R_b__n_grb(cos_phi, sin_phi, cos_the, sin_the, cos_psi, sin_psi):
    x_rot = np.array([[1,         0,           0],
                      [0,       cos_phi,   -sin_phi],
                      [0,       sin_phi,    cos_phi]])

    y_rot = np.array([[cos_the,      0,        sin_the],
                      [0,         1,           0],
                      [-sin_the,     0,       cos_the]])
    
    z_rot = np.array([[cos_psi,    -sin_psi,       0],
                      [sin_psi,     cos_psi,       0],
                      [0,          0,           1]])

    # World-to-body
    return z_rot @ y_rot @ x_rot

def R_b__n_cs(phi, theta, psi):
    x_rot = np.array([[1,         0,           0],
                      [0,       cs.cos(phi),   -cs.sin(phi)],
                      [0,       cs.sin(phi),    cs.cos(phi)]])

    y_rot = np.array([[cs.cos(theta),      0,        cs.sin(theta)],
                      [0,         1,           0],
                      [-cs.sin(theta),     0,        cs.cos(theta)]])
    
    z_rot = np.array([[cs.cos(psi),    -cs.sin(psi),       0],
                      [cs.sin(psi),     cs.cos(psi),       0],
                      [0,          0,           1]])

    # World-to-body
    return z_rot @ y_rot @ x_rot

def R_b__n_inv_grb(cos_phi, sin_phi, cos_the, sin_the, cos_psi, sin_psi):

    return np.array([[cos_psi*cos_the, cos_the*sin_psi, -sin_the],
                     [cos_psi*sin_phi*sin_the - cos_phi*sin_psi, cos_phi*cos_psi + sin_phi*sin_psi*sin_the, cos_the*sin_phi],
                     [sin_phi*sin_psi + cos_phi*cos_psi*sin_the, cos_phi*sin_psi*sin_the - cos_psi*sin_phi, cos_phi*cos_the]])

def R_b__n_inv_cs(phi, theta, psi):

    return np.array([[cs.cos(psi)*cs.cos(theta), cs.cos(theta)*cs.sin(psi), -cs.sin(theta)],
                     [cs.cos(psi)*cs.sin(phi)*cs.sin(theta) - cs.cos(phi)*cs.sin(psi), cs.cos(phi)*cs.cos(psi) + cs.sin(phi)*cs.sin(psi)*cs.sin(theta), cs.cos(theta)*cs.sin(phi)],
                     [cs.sin(phi)*cs.sin(psi) + cs.cos(phi)*cs.cos(psi)*cs.sin(theta), cs.cos(phi)*cs.sin(psi)*cs.sin(theta) - cs.cos(psi)*cs.sin(phi), cs.cos(phi)*cs.cos(theta)]])

def T(phi, theta):
    
    return np.array([[1,     np.sin(phi)*np.tan(theta),      np.cos(phi)*np.tan(theta)],
                     [0,          np.cos(phi),                   -np.sin(phi)],
                     [0,     np.sin(phi)/np.cos(theta),      np.cos(phi)/np.cos(theta)]])

def T_grb(cos_phi, sin_phi, one_over_cos_the, tan_the):
    
    return np.array([[1,     sin_phi*tan_the,      cos_phi*tan_the],
                     [0,          cos_phi,                   -sin_phi],
                     [0,     sin_phi*one_over_cos_the,      cos_phi*one_over_cos_the]])

def T_cs(phi, theta):
    
    return np.array([[1,     cs.sin(phi)*cs.tan(theta),      cs.cos(phi)*cs.tan(theta)],
                     [0,          cs.cos(phi),                   -cs.sin(phi)],
                     [0,     cs.sin(phi)/cs.cos(theta),      cs.cos(phi)/cs.cos(theta)]])
