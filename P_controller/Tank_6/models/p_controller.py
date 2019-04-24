class P_controller:
    def __init__(self, environment, AGENT_PARAMS, i):
        self.z_nom = AGENT_PARAMS["INIT_POSITION"]
        self.h_set = AGENT_PARAMS["SS_POSITION"] * environment.tanks[i].h
        self.Kc = AGENT_PARAMS["KC"]
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
