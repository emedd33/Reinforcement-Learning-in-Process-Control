from models.tank_model.tank import Tank
from visualize.window import Window
import numpy as np


class Environment:
    "Parameters are set in the params.py file"

    def __init__(self, TANK_PARAMS, TANK_DIST, MAIN_PARAMS):
        self.tanks = []
        for i, PARAMS in enumerate(TANK_PARAMS):
            tank = Tank(
                height=PARAMS["height"],
                radius=PARAMS["width"],
                max_level=PARAMS["max_level"],
                min_level=PARAMS["min_level"],
                pipe_radius=PARAMS["pipe_radius"],
                init_level=PARAMS["init_level"],
                dist=TANK_DIST[i],
            )
            self.tanks.append(tank)
        self.n_tanks = len(self.tanks)
        self.running = True
        self.terminated = [False] * self.n_tanks
        self.q_inn = [0] * (self.n_tanks + 1)
        self.show_rendering = MAIN_PARAMS["RENDER"]

        if self.show_rendering:
            self.window = Window(self.tanks)

    def get_next_state(self, z, state, t):
        """
        Calculates the dynamics of the agents action
        and gives back the next state
        """
        next_state = []
        prev_q_out = 0
        for i in range(self.n_tanks):
            dldt, prev_q_out = self.tanks[i].get_dhdt(z[i], t, prev_q_out)
            self.q_inn[i + 1] = prev_q_out
            self.tanks[i].change_level(dldt)
            z_ = 0 if i == 0 else z[i - 1]
            # Check terminate state
            if self.tanks[i].level < self.tanks[i].min:
                self.terminated[i] = True
                self.tanks[i].level = self.tanks[i].min
            elif self.tanks[i].level > self.tanks[i].max:
                self.terminated[i] = True
                self.tanks[i].level = self.tanks[i].max

            grad = (dldt + 0.1) / 0.2
            if self.tanks[i].level > 0.5 * self.tanks[i].h:
                above = 1
            else:
                above = 0

            next_state.append(
                np.array(
                    [self.tanks[i].level / self.tanks[i].h, grad, above, z_]
                )
            )
        return self.terminated, next_state

    def reset(self):
        "Reset the environment to the initial tank level and disturbance"
        init_state = []
        self.terminated = [False] * self.n_tanks
        for i in range(self.n_tanks):
            self.tanks[i].reset()  # reset to initial tank level
            if self.tanks[i].add_dist:
                self.tanks[i].dist.reset()  # reset to nominal disturbance
            init_state.append(
                np.array([self.tanks[i].init_l / self.tanks[i].h, 0, 1, 0])
            )  # Level plus gradient
        return [init_state], []

    def render(self, action):
        "Draw the water level of the tank in pygame"

        if self.show_rendering:
            running = self.window.Draw(action)
            if not running:
                self.running = False
