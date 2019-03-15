import numpy as np
from models.tank_model.disturbance import InflowDist


class Tank:
    "Cylindric tank"
    g = 9.81
    rho = 1000

    def __init__(
        self,
        height,
        radius,
        pipe_radius,
        max_level,
        min_level,
        init_level,
        dist,
        prev_tank=None,
    ):
        self.h = height
        self.r = radius
        self.A = radius ** 2 * np.pi

        self.level = height * init_level
        self.init_l = self.level

        self.max = max_level * height
        self.min = min_level * height
        self.prev_q_out = 0

        self.A_pipe = pipe_radius ** 2 * np.pi
        self.add_dist = dist["add"]
        if dist["add"]:
            self.dist = InflowDist(
                nom_flow=dist["nom_flow"],
                var_flow=dist["var_flow"],
                max_flow=dist["max_flow"],
                min_flow=dist["min_flow"],
                add_step=dist["add_step"],
                step_flow=dist["step_flow"],
                step_time=dist["step_time"],
                pre_def_dist=dist["pre_def_dist"],
                max_time=dist["max_time"],
            )

    def change_level(self, dldt):
        self.level += dldt * self.h

    def get_dhdt(self, action, t):
        "Calculates the change in water level"
        if self.add_dist:
            q_inn = self.dist.get_flow(t)
        else:
            q_inn = 0

        f, A_pipe, g, l, delta_p, rho, r = self.get_params(action)
        q_out = f * A_pipe * np.sqrt(1 * g * l + delta_p / rho)

        term1 = q_inn / (np.pi * r ** 2)
        term2 = (q_out) / (np.pi * r ** 2)
        return term1 - term2  # Eq: 1

    def get_valve(self, action):
        "linear valve equation"
        return action

    def get_params(self, action):
        "collects the tanks parameters"
        f = self.get_valve(action)
        return f, self.A_pipe, Tank.g, self.level, 0, Tank.rho, self.r
