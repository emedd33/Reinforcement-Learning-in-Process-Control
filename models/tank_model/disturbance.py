import numpy as np 
from params import ADD_INFLOW,DIST_DISTRIBUTION,DIST_NOM_FLOW,DIST_VARIANCE_FLOW,DIST_MAX_FLOW,DIST_MIN_FLOW


# Agent parameters
class InflowDist():

    def __init__(self, nom_flow=DIST_NOM_FLOW, variance=DIST_VARIANCE_FLOW, distribution=DIST_DISTRIBUTION):
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
