from models.tank_model.tank import Tank
from models.tank_model.disturbance import InflowDist
from visualize.window import Window
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np 
import math
from params import SS_POSITION,TANK_PARAMS,TANK_DIST,\
    TBCC,OBSERVATIONS,RENDER,LIVE_REWARD_PLOT,N_TANKS
class Environment():
    def __init__(self):
        
        self.tank = Tank(
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
        self.n_tanks = N_TANKS
        self.terminated = [False]*self.n_tanks

        self.show_rendering= RENDER
        self.live_plot = LIVE_REWARD_PLOT
        if RENDER:
            self.window = Window(self.tank)
        if LIVE_REWARD_PLOT:
            plt.ion()  # enable interactivity
            plt.figure(num="Rewards per episode")  # make a figure

    def get_next_state(self,z,state): 
        # models response to input change
        prev_q_out = 0
        next_state = []
    
        dldt,prev_q_out = self.tank.get_dhdt(z,prev_q_out) 
        self.tank.change_level(dldt)

        # Check terminate state
        if self.tank.l < self.tank.min:
            self.terminated = True
            self.tank.l = self.tank.min
        elif self.tank.l > self.tank.max:
            self.terminated = True
            self.tank.l = self.tank.max
        
        next_state = [self.tank.l/self.tank.h-SS_POSITION]        
        next_state = np.array(next_state)
        next_state = next_state.reshape(1,len(next_state))
        return self.terminated, next_state

            
    def reset(self):
        self.terminated = False
        self.tank.reset()
        state = []
        if self.tank.add_dist:
            self.tank.dist.reset() # reset to nominal disturbance
        init_state = [self.tank.init_l/self.tank.h-SS_POSITION] #Level plus gradient
        init_state = np.array(init_state)
        init_state = init_state.reshape(1,len(init_state))
        state.append(init_state)
        return state,[],TBCC,state

    def render(self,action):
        if RENDER:
            running = self.window.Draw(action)
            if not running:
                self.running = False

    def get_reward(self,state,terminated):
        if terminated:
            return -100
        # if state[0][0] > 0.45 and state[0][0] < 0.55:
        #     return 5
        # if state[0][0] > 0.4 and state[0][0] < 0.60:
        #     return 4
        # if state[0][0] > 0.3 and state[0][0] < 0.7:
        #     return 3
        # if state[0][0] > 0.2 and state[0][0] < 0.8:
        #     return 2
        # if state[0][0] > 0.1 and state[0][0] < 0.9:
        #     return 1
        else:
            return 1
        
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

