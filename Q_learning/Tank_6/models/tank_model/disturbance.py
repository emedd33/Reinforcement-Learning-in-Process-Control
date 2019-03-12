import numpy as np


class InflowDist:
    "Inlet disturbance flow"

    def __init__(
        self,
        nom_flow,
        var_flow,
        max_flow,
        min_flow,
        add_step,
        step_flow,
        step_time,
    ):
        self.var_flow = var_flow
        self.nom_flow = nom_flow
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.flow = nom_flow
        self.add_step = add_step
        if self.add_step:
            self.step_flow = step_flow
            self.step_time = step_time
        else:
            self.step_flow = self.flow
            self.step_time = 0

    def get_flow(self, t):
        "Gausian distribution of flow rate"
        if self.add_step:
            if t > self.step_time:
                self.flow = self.step_flow
                self.add_step = False
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
        self.add_step = True
