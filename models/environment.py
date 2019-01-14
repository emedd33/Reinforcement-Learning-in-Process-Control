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
        self.action_delay_counter = -OBSERVATIONS # Does not train on initial settings
        self.running = True
        self.episode = 0
        self.all_rewards = []
        
        self.show_rendering= RENDER
        self.live_plot = LIVE_REWARD_PLOT
        if RENDER:
            self.window = Window(self.model)
        if LIVE_REWARD_PLOT:
            plt.ion()  # enable interactivity
            fig = plt.figure(num="Max reward: {}".format(MAX_TIME))  # make a figure
            
            # def plot_rewards():
            #     plt.plot(rewards,label="Episode number: {}".format(e))
            #     plt.legend()

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
        
        # Check terminate state
        if self.model.l < self.model.min or self.model.l > self.model.max:
            return "Terminated"
        return self.model.l
            
    def reset(self):
        self.model.reset() # reset to initial tank level
        self.dist.reset() # reset to nominal disturbance
        init_action = 0.5
        return None,[0.5*TANK_HEIGHT], [],-OBSERVATIONS, init_action

    def render(self,action,next_state):
        if RENDER:
            running = self.window.Draw(action,next_state)
            if not running:
                self.running = False
    def get_reward(self,state):
        if state == "Terminated":
            return -TANK_HEIGHT**2
        reward = -(state - SS_POSITION)**2 # MSE
        return reward
    def plot_rewards(self):
        plt.plot(self.all_rewards,label="Episode number: {}".format(self.episode))
        plt.legend()

    def plot(self,all_rewards,episode):
        self.all_rewards.extend(all_rewards)
        self.episode = episode
        drawnow(self.plot_rewards)

