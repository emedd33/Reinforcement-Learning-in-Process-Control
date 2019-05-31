MAIN_PARAMS = {"EPISODES": 1, "MAX_TIME": 200, "RENDER": True}

# Agent parameters
AGENT1_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 200,
}

AGENT2_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 250,
}

TANK1_PARAMS = {
    "height": 10,
    "init_level": 0.5,
    "width": 10,
    "pipe_radius": 0.5,
    "max_level": 0.75,
    "min_level": 0.25,
}

TANK1_DIST = {
    "add": True,
    "pre_def_dist": True,
    "nom_flow": 1,  # 2.7503
    "var_flow": 0.1,
    "max_flow": 2,
    "min_flow": 0.7,
    "step_flow": 2,
    "add_step": False,
    "max_time": MAIN_PARAMS["MAX_TIME"],
    "step_time": int(MAIN_PARAMS["MAX_TIME"] / 2),
}

TANK2_PARAMS = {
    "height": 10,
    "init_level": 0.5,
    "width": 8,
    "pipe_radius": 0.5,
    "max_level": 0.75,
    "min_level": 0.25,
}

TANK2_DIST = {
    "add": False,
    "nom_flow": 1,  # 2.7503
    "var_flow": 0.1,
    "max_flow": 2,
    "min_flow": 0.7,
    "add_step": False,
    "step_time": int(MAIN_PARAMS["MAX_TIME"] / 2),
    "step_flow": 2,
    "max_time": MAIN_PARAMS["MAX_TIME"],
}
AGENT_PARAMS_LIST = [AGENT1_PARAMS, AGENT2_PARAMS]
TANK_PARAMS_LIST = [TANK1_PARAMS, TANK2_PARAMS]
TANK_DIST_LIST = [TANK1_DIST, TANK2_DIST]
