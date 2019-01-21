import numpy as np 
import pygame
from params import MAX_LEVEL,MIN_LEVEL,TANK_HEIGHT,TANK_RADIUS,INIT_LEVEL,TANK_PIPE_RADIUS
class Tank(): # Cylindric tank
    def __init__(self, 
    height=TANK_HEIGHT, 
    radius=TANK_RADIUS, 
    level=INIT_LEVEL, # %
    max_level=MAX_LEVEL, 
    min_level=MIN_LEVEL, 
    rho=1000,
    pipe_radius=TANK_PIPE_RADIUS #m
    ):
        self.h = height
        self.r = radius
        self.A = radius**2*np.pi

        self.l = height*level
        self.init_l = self.l
        
        self.max = height*max_level
        self.min = height*min_level
        self.rho = rho
        self.g = 9.81
        self.A_pipe = pipe_radius**2*np.pi

    def get_dl_outflow(self,z,p_out=1): # Z is the choke opening
        v_out = np.sqrt(2*(self.g*self.l-p_out/self.rho)) #bernoulli
        q_out = v_out*self.A_pipe*z
        dl = -q_out/(np.pi * self.r**2) 
        return dl

    def get_dl_inflow(self,q_inn):
        dl = q_inn/(self.A*self.rho)
        return dl

    def change_level(self,dldt):
        self.l += dldt

    def reset(self):
        self.l = self.init_l

    def get_valve(self,action):
        return action
        
    def get_params(self,action):
        f = self.get_valve(action)
        return f,self.A_pipe,self.g,self.l,0,self.rho,self.r
        
        
    
