import numpy as np 


class InflowDist():
    "Inlet disturbance flow"
    
    def __init__(self, nom_flow, var_flow,max_flow,min_flow):
        self.var_flow = var_flow
        self.nom_flow = nom_flow
        self.min_flow = min_flow
        self.max_flow = max_flow

    def get_flow(self):
        "Gausian distribution of flow rate"
        
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
        "Sets dstubance flow to nominal value"
        self.flow = self.nom_flow
