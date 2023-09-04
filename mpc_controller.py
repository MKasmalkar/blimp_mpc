from ExampleMPC import ExampleMPC
import socket
import numpy as np
import struct
import sys
import time

print("MPC CONTROLLER")
print()

controller = ExampleMPC('dynamics', 3)

HOST = '127.0.0.1'
PORT = 8000

DATA_SIZE = 16

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    conn, addr = s.accept()
    
    with conn:
        
        try:
            while True:
                data = conn.recv(DATA_SIZE)
                if not data:
                    continue
                
                x0, x1 = struct.unpack('<2f', data)

                x = np.matrix([[x0],
                                [x1]])

                u = controller.get_control_vector(x)
                conn.sendall(struct.pack('<1f', float(u[0])))

        except KeyboardInterrupt:
            print("Done!")