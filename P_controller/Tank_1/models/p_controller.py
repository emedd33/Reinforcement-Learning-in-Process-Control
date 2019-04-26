import numpy as np


class P_controller:
    def __init__(self, environment, AGENT_PARAMS, i):
        self.z_nom = AGENT_PARAMS["INIT_POSITION"]
        self.tank = environment.tanks[i]
        self.h_set = AGENT_PARAMS["SS_POSITION"] * environment.tanks[i].h
        self.k = self.tank.init_l / self.z_nom
        self.tau1 = (np.pi * self.tank.r) / (
            self.tank.init_l * self.tank.A_pipe * 2 * 9.81
        )
        self.tau_c = AGENT_PARAMS["TAU_C"]
        self.evalv_kc(self.tau_c)
        self.action_deley = AGENT_PARAMS["ACTION_DELAY"]
        self.action = AGENT_PARAMS["INIT_POSITION"]
        self.action_buffer = 99999

    def get_z(self, h):
        if self.action_buffer > self.action_deley:
            delta_h = h - self.h_set
            z = delta_h * self.Kc + self.z_nom
            z = 1 if z > 1 else z
            z = 0 if z < 0 else z
            self.action = z
            self.action_buffer = 0
        else:
            self.action_buffer += 1
        return self.action

    def evalv_kc(self, tau_c):
        self.Kc = self.k / (self.tau1 * tau_c)
