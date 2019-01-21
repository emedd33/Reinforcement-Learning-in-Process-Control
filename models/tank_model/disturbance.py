import numpy as np 
from params import ADD_INFLOW,DIST_DISTRIBUTION,DIST_PIPE_RADIUS,DIST_NOM_FLOW,DIST_VARIANCE_FLOW,DIST_MAX_FLOW,DIST_MIN_FLOW


# Agent parameters
class InflowDist():

    def __init__(self, pipe_r=DIST_PIPE_RADIUS, nom_flow=DIST_NOM_FLOW, variance=DIST_VARIANCE_FLOW, distribution=DIST_DISTRIBUTION):
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
