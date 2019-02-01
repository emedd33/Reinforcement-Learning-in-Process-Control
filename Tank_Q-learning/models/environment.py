from models.tank_model.tank import Tank
from models.tank_model.disturbance import InflowDist
from visualize.window import Window
from params import *
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np 
class Environment():
    def __init__(self):
        
        self.model = Tank(
            height=TANK_PARAMS['height'],
            radius=TANK_PARAMS['width'],
            max_level=TANK_PARAMS['max_level'],
            min_level=TANK_PARAMS['min_level'],
            pipe_radius=TANK_PARAMS['pipe_radius'],
            dist = TANK_DIST
            ) 
        
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
        if self.model.add_dist:
            q_inn = self.dist.get_flow()
        else:
            q_inn = 0
        f,A_pipe,g,l,delta_p,rho,r = self.model.get_params(action) 
        
        q_out = f*A_pipe*np.sqrt(1*g*l+delta_p/rho)
        term1 = q_inn/(np.pi*r**2)
        term2 = (q_out)/(np.pi*r**2)
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
        grad = self.model.l/self.model.h - state[0][0]
        next_state = np.array([self.model.l/self.model.h,grad])
        next_state = next_state.reshape(1, state.size)
        return self.terminated, next_state

            
    def reset(self):
        self.model.reset() # reset to initial tank level
        if self.model.add_dist:
            self.model.dist.reset()
        self.terminated = False
        # init_state = OBSERVATIONS*[self.model.init_l/self.model.h]
        init_state = [self.model.init_l/self.model.h,0] 
        init_state = np.array([init_state])
        return init_state,TBCC,init_state,[]

    def render(self,action):
        if RENDER:
            running = self.window.Draw(action)
            if not running:
                self.running = False

    def get_reward(self,state,terminated):
        if terminated:
            return -10
        elif state[0][0] > 0.25 and state[0][0] < 0.75:
            return 1
        else:
            return 0
        
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

