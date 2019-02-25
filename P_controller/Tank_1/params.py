MAIN_PARAMS = {
    "Episodes": 1,
    "Mean_episodes": 20,
    "Max_time": 200,
    "RENDER": False,
    "LIVE_REWARD_PLOT": False,
}

# Agent parameters
AGENT_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "VALVE_START_POSITION": 0.5,
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.5,
    "KC": 0.16,
}

TANK_PARAMS = {
    "height": 10,
    "init_level": 0.5,
    "width": 3,
    "pipe_radius": 0.5,
    "max_level": 0.9,
    "min_level": 0.1,
}

TANK_DIST = {
    "add": True,
    "nom_flow": 1,  # 2.7503
    "var_flow": 0,
    "max_flow": 2,
    "min_flow": 0.7,
    "add_step": True,
    "step_time": int(MAIN_PARAMS["Max_time"] / 2),
    "step_flow": 2,
}
