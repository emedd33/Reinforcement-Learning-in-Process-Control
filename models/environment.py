from models.tank_model.tank import Tank
from models.tank_model.disturbance import InflowDist
from visualize.window import Window
from params import *
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np 
class Environment():
    def __init__(self):
        self.model = Tank(TANK_HEIGHT,TANK_RADIUS) # get model
        self.dist = InflowDist(DIST_PIPE_RADIUS,DIST_NOM_FLOW,DIST_VARIANCE_FLOW)
        self.add_dist = ADD_INFLOW
        self.action_delay= TBCC
        self.action_delay_counter = TBCC
        self.running = True
        self.episode = 0
        self.rewards = []
        
        self.is_terminate_state = False

        self.show_rendering= RENDER
        self.live_plot = LIVE_REWARD_PLOT
        if RENDER:
            self.window = Window(self.model)
        if LIVE_REWARD_PLOT:
            plt.ion()  # enable interactivity
            fig = plt.figure(num="Max reward: {}".format(MAX_TIME))  # make a figure
            
            def plot_rewards():
                plt.plot(rewards,label="Episode number: {}".format(e))
                plt.legend()

    def remember(self,action,state):
        self.remember.append([action,state])

    def get_next_state(self,action):
        # models response to input change
        dl = self.model.get_dl_outflow(z=action)
        self.model.change_level(dl)

        # Random disturbance
        if ADD_INFLOW:
            q_inn = self.dist.get_flow()
            q_inn = 0 if q_inn < 0 else q_inn
            q_inn = DIST_MAX_FLOW if q_inn > DIST_MAX_FLOW else q_inn
            dl_dist = self.model.get_dl_inflow(q_inn)
            self.model.change_level(dl_dist)    
        
        self.action_delay_counter +=1
        # Check terminate state
        if self.model.l < self.model.min or self.model.l > self.model.max:
            return "Terminated"
        return self.model.l
            
    def reset(self):
        self.model.reset() # reset to initial tank level
        self.dist.reset() # reset to nominal disturbance

    def render(self,action,next_state):
        if RENDER:
            running = self.window.Draw(action,next_state)
            if not running:
                self.running = False
    def get_reward(self):
        reward = np.abs(self.model.l - SS_POSITION) # Divergent from set point
        if self.is_terminate_state:
            reward *= 3 
        return reward
    def plot_rewards(self):
        plt.plot(self.rewards,label="Episode number: {}".format(self.episode))
        plt.legend()

    def plot(self,rewards,episode):
        self.rewards.append(rewards)
        self.episode = episode
        drawnow(self.plot_rewards)

