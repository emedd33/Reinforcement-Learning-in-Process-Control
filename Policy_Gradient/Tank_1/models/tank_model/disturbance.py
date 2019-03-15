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
        pre_def_dist,
        max_time,
    ):
        self.var_flow = var_flow
        self.nom_flow = nom_flow
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.max_time = max_time
        self.flow = [nom_flow]
        self.add_step = add_step
        if self.add_step:
            self.step_flow = step_flow
            self.step_time = step_time
        self.pre_def_dist = pre_def_dist
        if self.pre_def_dist:
            csv_name = "disturbance_" + str(self.max_time) + ".csv"
            self.flow = np.genfromtxt(csv_name, delimiter=",")

    def get_flow(self, t):
        "Gausian distribution of flow rate"
        if self.pre_def_dist:
            return self.flow[t]
        else:
            if self.add_step:
                if t > self.step_time:
                    self.flow.append(self.step_flow)
                    self.max_flow = self.step_flow
                    self.add_step = False
            new_flow = np.random.normal(self.flow[-1], self.var_flow)
            if new_flow > self.max_flow:
                self.flow.append(self.max_flow)
                return self.flow[-1]
            elif new_flow < self.min_flow:
                self.flow.append(self.min_flow)
                return self.flow[-1]
            else:
                self.flow.append(new_flow)
                return self.flow[-1]

    def reset(self):
        "Sets dstubance flow to nominal value"
        if not self.pre_def_dist:
            self.flow = [self.nom_flow]
