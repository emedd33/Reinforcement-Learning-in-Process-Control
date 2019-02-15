from params import VALVE_START_POSITION,SS_POSITION
import numpy as np 
class P_controller():
    def __init__(self,environment):
        self.z_nom = VALVE_START_POSITION
        self.A = environment.model.r**2*np.pi
        self.h_set = SS_POSITION*environment.model.h
        self.Kc = 0.5#(-2*self.A)/self.z_nom
        
    def get_z(self,h):
        delta_h = h-self.h_set
        z = delta_h*self.Kc+self.z_nom
        z = 1 if z > 1 else z
        z = 0 if z < 0 else z
        return z