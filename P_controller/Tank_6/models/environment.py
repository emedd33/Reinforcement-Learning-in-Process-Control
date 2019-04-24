from models.tank_model.tank import Tank
from visualize.window import Window


class Environment:
    "Parameters are set in the params.py file"

    def __init__(self, TANK_PARAMS_LIST, TANK_DIST_LIST, MAIN_PARAMS):
        self.tanks = []
        for i, PARAMS in enumerate(TANK_PARAMS_LIST):
            tank = Tank(
                height=PARAMS["height"],
                radius=PARAMS["width"],
                max_level=PARAMS["max_level"],
                min_level=PARAMS["min_level"],
                pipe_radius=PARAMS["pipe_radius"],
                init_level=PARAMS["init_level"],
                dist=TANK_DIST_LIST[i],
            )
            self.tanks.append(tank)
        self.running = True
        self.episode = 0
        self.terminated = False

        self.show_rendering = MAIN_PARAMS["RENDER"]

        if self.show_rendering:
            self.window = Window(self.tanks)

    def get_next_state(self, z, i, t, q_out):
        """
        Calculates the dynamics of the agents action and
        gives back the next state
        """
        dldt, q_out = self.tanks[i].get_dhdt(z, t, q_out)
        self.tanks[i].change_level(dldt)
        # Check terminate state
        if self.tanks[i].level < self.tanks[i].min:
            self.terminated = True
            self.tanks[i].level = self.tanks[i].min
        elif self.tanks[i].level > self.tanks[i].max:
            self.terminated = True
            self.tanks[i].level = self.tanks[i].max
        return self.tanks[i].level, q_out

    def render(self, action):
        "Draw the water level of the tank in pygame"

        if self.render:
            running = self.window.Draw(action)
            if not running:
                self.running = False
