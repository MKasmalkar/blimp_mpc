from scipy.spatial.transform import Rotation as R
import numpy as np

def quat2euler(q):
    r = R.from_quat(q)
    return r.as_euler('XYZ', degrees=False)

def euler2quat(e):
    r = R.from_euler('xyz', e.T)
    return r.as_quat()

def rotation_matrix(phi, theta, psi):
    # roll (about x)
    # pitch (about y)
    # yaw (about psi)
    # GNC standard convention: principle rotations in zyx order.
    return np.array([[np.cos(psi) * np.cos(theta), -np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi),
                      np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi) * np.sin(theta)],
                      [np.sin(psi) * np.cos(theta), np.cos(psi) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
                       -np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi) * np.cos(phi)],
                       [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]])

def transformation_matrix(phi, theta, psi):
    return np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])