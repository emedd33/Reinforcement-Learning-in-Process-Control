import numpy as np 
import pygame
from params import INIT_LEVEL
from models.tank_model.disturbance import InflowDist
class Tank(): # Cylindric tank
    g=9.81
    rho=1000
    
    def __init__(self, 
        height, 
        radius, 
        pipe_radius, 
        max_level, 
        min_level, 
        dist,
        prev_tank=None,
        level=INIT_LEVEL, # %  
    ):
        self.h = height
        self.r = radius
        self.A = radius**2*np.pi

        self.l = height*level
        self.init_l = self.l
        
        self.max = max_level*height
        self.min = min_level*height
        self.prev_q_out = 0
        # if prev_tank is not None:
        #     self.prev_tank_q_out = prev_tank.get_nom_q_out()
        self.A_pipe = pipe_radius**2*np.pi
        self.add_dist = dist['add']
        if dist['add']:
            self.dist =  InflowDist(
                nom_flow=dist['nom_flow'],
                var_flow=dist['var_flow'],    
                max_flow=dist['max_flow'],
                min_flow=dist['min_flow'],
            )

    # def get_dl_outflow(self,z,p_out=1): # Z is the choke opening
    #     v_out = np.sqrt(2*(Tank.g*self.l-p_out/Tank.rho)) #bernoulli
    #     q_out = v_out*self.A_pipe*z
    #     dl = -q_out/(np.pi * self.r**2) 
    #     return dl

    # def get_dl_inflow(self,q_inn):
    #     dl = q_inn/(self.A*Tank.rho)
    #     return dl

    def change_level(self,dldt):
        self.l += dldt*self.h

    def reset(self):
        self.l = self.init_l

    def get_valve(self,action): #
        return action
        
    def get_params(self,action):
        f = self.get_valve(action)
        return f,self.A_pipe,Tank.g,self.l,0,Tank.rho,self.r
        
        
    
