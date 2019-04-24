from models.tank_model.tank import Tank
from visualize.window import Window
import matplotlib.pyplot as plt

class Environment:
    "Parameters are set in the params.py file"

    def __init__(self, TANK_PARAMS_LIST, TANK_DIST_LIST, MAIN_PARAMS):
        self.model = []
        for i, TANK_PARAMS in enumerate(TANK_PARAMS_LIST):
            tank = Tank(
                height=TANK_PARAMS["height"],
                radius=TANK_PARAMS["width"],
                max_level=TANK_PARAMS["max_level"],
                min_level=TANK_PARAMS["min_level"],
                pipe_radius=TANK_PARAMS["pipe_radius"],
                init_level=TANK_PARAMS["init_level"],
                dist=TANK_DIST_LIST[i],
            )
            self.model.append(tank)

        self.running = True
        self.episode = 0
        self.all_rewards = []
        self.terminated = False

        self.show_rendering = MAIN_PARAMS["RENDER"]
        self.live_plot = MAIN_PARAMS["LIVE_REWARD_PLOT"]

        if self.show_rendering:
            self.window = Window(self.model)
        if self.live_plot:
            plt.ion()  # enable interactivity
            plt.figure(num="Rewards per episode")  # make a figure

    def get_next_state(self, z, i, t, q_out):
        """
        Calculates the dynamics of the agents action and
        gives back the next state
        """
        dldt, q_out = self.model[i].get_dhdt(z, t, q_out)
        self.model[i].change_level(dldt)
        # Check terminate state
        if self.model[i].level < self.model[i].min:
            self.terminated = True
            self.model[i].level = self.model[i].min
        elif self.model[i].level > self.model[i].max:
            self.terminated = True
            self.model[i].level = self.model[i].max
        return self.model[i].level, q_out

    def render(self, action):
        "Draw the water level of the tank in pygame"

        if self.render:
            running = self.window.Draw(action)
            if not running:
                self.running = False

    def get_reward(self, h):
        h = h / self.model.h
        reward = (h - 0.5) ** 2
        return reward
        if h > 0.49 and h < 0.51:
            return 5
        if h > 0.45 and h < 0.55:
            return 4
        if h > 0.4 and h < 0.6:
            return 3
        if h > 0.3 and h < 0.7:
            return 2
        if h > 0.2 and h < 0.8:
            return 1
        else:
            return 0

