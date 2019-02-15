from models.tank_model.tank import Tank
from models.tank_model.disturbance import InflowDist
from visualize.window import Window
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np 
from params import SS_POSITION,TANK_PARAMS,TANK_DIST,\
    OBSERVATIONS,RENDER,LIVE_REWARD_PLOT
class Environment():
    "Parameters are set in the params.py file"
    
    def __init__(self):
        
        self.model = Tank(
            height=TANK_PARAMS['height'],
            radius=TANK_PARAMS['width'],
            max_level=TANK_PARAMS['max_level'],
            min_level=TANK_PARAMS['min_level'],
            pipe_radius=TANK_PARAMS['pipe_radius'],
            dist = TANK_DIST
            ) 
        
        self.running = True
        self.episode = 0
        self.all_rewards = []
        self.terminated = False

        self.show_rendering= RENDER
        self.live_plot = LIVE_REWARD_PLOT
        
        if RENDER:
            self.window = Window(self.model)
        if LIVE_REWARD_PLOT:
            plt.ion()  # enable interactivity
            plt.figure(num="Rewards per episode")  # make a figure

    def get_next_state(self,z): 
        "Calculates the dynamics of the agents action and gives back the next state"
    
        dldt = self.model.get_dhdt(z) 
        self.model.change_level(dldt)
        # Check terminate state
        if self.model.l < self.model.min:
            self.terminated = True
            self.model.l = self.model.min
        elif self.model.l > self.model.max:
            self.terminated = True
            self.model.l = self.model.max
        return self.model.l
        
            
    def reset(self):
        "Reset the environment to the initial tank level and disturbance"

        state = []
        self.terminated = False
        self.model.reset() # reset to initial tank level
        if self.model.add_dist:
            self.model.dist.reset() # reset to nominal disturbance
        init_state = [self.model.init_l/self.model.h,0] #Level plus gradient
        state.append(init_state)
        state = np.array(state)
        state = state.reshape(1,2)
        return state,state,[]

    def render(self,action):
        "Draw the water level of the tank in pygame"

        if RENDER:
            running = self.window.Draw(action)
            if not running:
                self.running = False

    def get_reward(self,state,terminated):
        "Calculates the environments reward for the next state"

        if terminated:
            reward=-10
        if state[0][0] > 0.25 and state[0][1] < 0.75:
            reward=1
        else:
            reward=0
        return reward
        
    def plot_rewards(self):
        "drawnow plot of the reward"

        plt.plot(self.all_rewards,label="Exploration rate: {} %".format(self.epsilon*100))
        plt.legend()

    def plot(self,all_rewards,epsilon):
        "Live plot of the reward"
        self.all_rewards = all_rewards
        self.epsilon = round(epsilon,4)
        try:
            drawnow(self.plot_rewards)
        except:
            print("Break")

