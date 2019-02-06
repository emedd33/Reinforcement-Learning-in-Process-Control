import numpy as np 



# Agent parameters
class InflowDist():

    def __init__(self, nom_flow, var_flow,max_flow,min_flow):
        self.var_flow = var_flow
        self.nom_flow = nom_flow
        self.min_flow = min_flow
        self.max_flow = max_flow

    def get_flow(self):
        
        new_flow = np.random.normal(self.flow, self.var_flow)
        if new_flow > self.max_flow:
            self.flow = self.max_flow
            return self.flow
        elif new_flow < self.min_flow:
            self.flow = self.min_flow
            return self.flow
        else:
            self.flow = new_flow
            return self.flow
            
    def reset(self):
        self.flow = self.nom_flow
