import numpy as np 
import pygame

class Tank(): # Cylindric tank
    def __init__(self, 
    height=10, 
    radius=3, 
    level=0.5, 
    max_level=0.99, 
    min_level=0.01, 
    rho=1000,
    pipe_radius=0.2
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

    def change_level(self,dl):
        self.l += dl

    def reset(self):
        self.l = self.init_l

        
        
    
