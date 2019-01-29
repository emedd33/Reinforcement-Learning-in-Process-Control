from models.tank_model.tank import Tank
from models.tank_model.disturbance import InflowDist
from visualize.window import Window
from params import *
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np 
from params import SS_POSITION
class Environment():
    def __init__(self):
        self.model = Tank() # get tank model
        if ADD_INFLOW:
            self.dist = InflowDist()
        self.action_delay= TBCC
        self.action_delay_counter = -OBSERVATIONS # Does not train on initial settings
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
            fig = plt.figure(num="Rewards per episode")  # make a figure

    def get_dhdt(self,action):
        if ADD_INFLOW:
            q_inn = self.dist.get_flow()
            q_inn = DIST_MIN_FLOW if q_inn < DIST_MIN_FLOW else q_inn
            q_inn = DIST_MAX_FLOW if q_inn > DIST_MAX_FLOW else q_inn
        else:
            q_inn = 0
        f,A_pipe,g,l,delta_p,rho,r = self.model.get_params(action) 
        
        term1 = q_inn/(np.pi*r**2)
        term2 = (f*A_pipe*np.sqrt(1*g*l+delta_p/rho))/(np.pi*r**2)
        return term1- term2 # Eq: 1

    def get_next_state(self,action,state): 
        # models response to input change
        dldt = self.get_dhdt(action)
        self.model.change_level(dldt)

        # Check terminate state
        if self.model.l < self.model.min:
            self.terminated = True
            self.model.l = self.model.min
        elif self.model.l > self.model.max:
            self.terminated = True
            self.model.l = self.model.max
        next_state = np.append(state[0][1:],self.model.l/self.model.h)
        next_state = next_state.reshape(1, state.size)
        return self.terminated, next_state

            
    def reset(self):
        self.model.reset() # reset to initial tank level
        if ADD_INFLOW:
            self.dist.reset() # reset to nominal disturbance
        self.terminated = False
        init_state = OBSERVATIONS*[self.model.init_l/self.model.h]
        init_state = np.array([init_state])
        return init_state,TBCC,init_state,[]

    def render(self,action,next_state):
        if RENDER:
            running = self.window.Draw(action,next_state)
            if not running:
                self.running = False

    def get_reward(self,state,terminated,t):
        if state[0][-1] > 0.25 and state[0][-1] < 0.75:
            return 1
        if terminated: # sums up the rest of the episode time
            return -10
        else:
            return -1
        
    def plot_rewards(self):
        plt.plot(self.all_rewards,label="Exploration rate: {} %".format(self.epsilon*100))
        plt.legend()

    def plot(self,all_rewards,epsilon):
        self.all_rewards = all_rewards
        self.epsilon = round(epsilon,4)
        try:
            drawnow(self.plot_rewards)
        except:
            print("Break")

