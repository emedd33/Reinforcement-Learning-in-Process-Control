import numpy as np 
import pygame
import math
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
        
        self.A_pipe = pipe_radius**2*np.pi
        self.add_dist = dist['add']
        if dist['add']:
            self.dist =  InflowDist(
                nom_flow=dist['nom_flow'],
                var_flow=dist['var_flow'],    
                max_flow=dist['max_flow'],
                min_flow=dist['min_flow'],
            )


    def change_level(self,dldt):
        self.l += dldt*self.h

    def get_dhdt(self,action,prev_q_out):
        if self.add_dist:
            q_inn = self.dist.get_flow() + self.prev_q_out
        else:
            q_inn = self.prev_q_out
        self.prev_q_out = prev_q_out

        f,A_pipe,g,l,delta_p,rho,r = self.get_params(action) 
        q_out = f*A_pipe*np.sqrt(1*g*l+delta_p/rho)

        term1 = q_inn/(np.pi*r**2)
        term2 = (q_out)/(np.pi*r**2)
        return term1- term2,q_out # Eq: 1

    def reset(self):
        self.l = self.init_l

    def get_valve(self,action): #
        return action
        
    def get_params(self,action):
        f = self.get_valve(action)
        return f,self.A_pipe,Tank.g,self.l,0,Tank.rho,self.r
        
        
    
