

class PID:
    def __init__(self, Kp, Ki, Kd, dT):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dT = dT

        self.prev_error = 0
        self.error_integral = 0

    def get_ctrl(self, error):
        self.error_integral += error*self.dT

        self.derivative = (error - self.prev_error)/self.dT
        self.prev_error = error

        return self.Kp*error + self.Ki*self.error_integral + self.Kd*self.derivative