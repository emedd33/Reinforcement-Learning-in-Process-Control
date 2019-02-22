from models.tank_model.tank import Tank
from visualize.window import Window
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np


class Environment:
    "Parameters are set in the params.py file"

    def __init__(self, TANK_PARAMS, TANK_DIST, MAIN_PARAMS):

        self.tank = Tank(
            height=TANK_PARAMS["height"],
            radius=TANK_PARAMS["width"],
            max_level=TANK_PARAMS["max_level"],
            min_level=TANK_PARAMS["min_level"],
            pipe_radius=TANK_PARAMS["pipe_radius"],
            init_level=TANK_PARAMS["init_level"],
            dist=TANK_DIST,
        )

        self.running = True
        self.episode = 0
        self.all_rewards = []
        self.terminated = False

        self.show_rendering = MAIN_PARAMS["RENDER"]
        self.live_plot = MAIN_PARAMS["LIVE_REWARD_PLOT"]

        if self.show_rendering:
            self.window = Window(self.tank)
        if self.live_plot:
            plt.ion()  # enable interactivity
            plt.figure(num="Rewards per episode")  # make a figure

    def get_next_state(self, z, state):
        """
        Calculates the dynamics of the agents action
        and gives back the next state
        """

        dldt = self.tank.get_dhdt(z)
        self.tank.change_level(dldt)

        # Check terminate state
        if self.tank.level < self.tank.min:
            self.terminated = True
            self.tank.level = self.tank.min
        elif self.tank.level > self.tank.max:
            self.terminated = True
            self.tank.level = self.tank.max
        # grad = (dldt + 1) / 2
        next_state = np.array([self.tank.level / self.tank.h])
        # next_state = next_state.reshape(1,2)
        return self.terminated, next_state

    def reset(self):
        "Reset the environment to the initial tank level and disturbance"

        state = []
        self.terminated = False
        self.tank.reset()  # reset to initial tank level
        if self.tank.add_dist:
            self.tank.dist.reset()  # reset to nominal disturbance
        init_state = [self.tank.init_l / self.tank.h]  # Level plus gradient
        # state.append(init_state)
        state = np.array(init_state)
        # state = state.reshape(1,2)
        states = [state]
        return states, []

    def render(self, action):
        "Draw the water level of the tank in pygame"

        if self.show_rendering:
            running = self.window.Draw(action)
            if not running:
                self.running = False

    def get_reward_1(self, state, terminated):
        "Calculates the environments reward for the next state"

        if terminated:
            return -10
        if state[0] > 0.25 and state[0] < 0.75:
            return 1
        return 0

    def plot_rewards(self):
        "drawnow plot of the reward"

        plt.plot(
            self.all_rewards,
            label="Exploration rate: {} %".format(self.epsilon * 100),
        )
        plt.legend()

    def plot(self, all_rewards, epsilon):
        "Live plot of the reward"
        self.all_rewards = all_rewards
        self.epsilon = round(epsilon, 4)
        try:
            drawnow(self.plot_rewards)
        except KeyboardInterrupt:
            print("Break")
