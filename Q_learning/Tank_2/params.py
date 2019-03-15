MAIN_PARAMS = {
    "EPISODES": 10000,
    "MEAN_EPISODE": 10,
    "MAX_TIME": 200,
    "RENDER": False,
    "LIVE_REWARD_PLOT": False,
}

AGENT_PARAMS = {
    "N_TANKS": 2,
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.5,
    "ACTION_DELAY": [5, 4],
    "INIT_ACTION": 0,
    "EPSILON_MIN": 0,
    "EPSILON_DECAY": [0.95, 0.98],
    "LEARNING_RATE": 0.001,
    "HIDDEN_LAYER_SIZE": [10],
    "BATCH_SIZE": 5,
    "MEMORY_LENGTH": 10000,
    "OBSERVATIONS": 4,  # level, gradient, is_above 0.5, prevous valve position
    "VALVE_POSITIONS": 3,
    "GAMMA": 0.9,
    "EPSILON": 1,
    "SAVE_MODEL": True,
    "LOAD_MODEL": False,
    "TRAIN_MODEL": True,
    "MODEL_NAME": "Network_[5]HL108",
}
AGENT_PARAMS["BUFFER_THRESH"] = AGENT_PARAMS["BATCH_SIZE"] * 1

# Model parameters Tank 1
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
    "pre_def_dist": False,
    "nom_flow": 1,  # 2.7503
    "var_flow": 0.05,
    "max_flow": 1.5,
    "min_flow": 0.5,
    "add_step": False,
    "step_time": int(MAIN_PARAMS["MAX_TIME"] / 2),
    "step_flow": 2,
    "max_time": MAIN_PARAMS["MAX_TIME"],
}
# Model parameters Tank 1
TANK2_PARAMS = {
    "height": 10,
    "init_level": 0.5,
    "width": 10,
    "pipe_radius": 0.5,
    "max_level": 0.75,
    "min_level": 0.25,
}
TANK2_DIST = {
    "add": False,
    "pre_def_dist": False,
    "nom_flow": 1,  # 2.7503
    "var_flow": 0.05,
    "max_flow": 2,
    "min_flow": 0.5,
    "add_step": False,
    "step_time": int(MAIN_PARAMS["MAX_TIME"] / 2),
    "step_flow": 2,
    "max_time": MAIN_PARAMS["MAX_TIME"],
}
TANK_PARAMS = [TANK1_PARAMS, TANK2_PARAMS]
TANK_DIST = [TANK1_DIST, TANK2_DIST]
