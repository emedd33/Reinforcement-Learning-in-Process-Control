import numpy as np 

class InflowDist():

    def __init__(self, pipe_r, nom_flow=3, variance=1, distribution="gauss"):
        self.A = pipe_r**2*np.pi 
        self.expected = nom_flow
        self.variance = variance
        self.nom_flow = nom_flow
        self.flow = nom_flow
        self.dist = distribution

    def get_flow(self):
        if self.dist is "gauss":
            new_flow = np.random.normal(self.flow, self.variance)
            self.flow = new_flow
            return self.flow
    def reset(self):
        self.flow = self.nom_flow
