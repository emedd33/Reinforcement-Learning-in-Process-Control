MAIN_PARAMS = {
    "EPISODES": 2000,
    "MEAN_EPISODE": 10,
    "MAX_TIME": 200,
    "RENDER": True,
    "LIVE_REWARD_PLOT": False,
}

AGENT_PARAMS = {
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.5,
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.5,
    "EPSILON_MIN": 0,
    "EPSILON_DECAY": 0.95,
    "LEARNING_RATE": 0.001,
    "HIDDEN_LAYER_SIZE": [10],
    "BATCH_SIZE": 100,
    "MEMORY_LENGTH": 10000,
    "OBSERVATIONS": 1,
    "VALVE_POSITIONS": 15,
    "GAMMA": 0.9,
    "EPSILON": 0,
    "SAVE_MODEL": False,
    "LOAD_MODEL": True,
    "TRAIN_MODEL": False,
}
AGENT_PARAMS["BUFFER_THRESH"] = AGENT_PARAMS["BATCH_SIZE"] * 1

TANK_PARAMS = {
    "height": 10,
    "init_level": 0.5,
    "width": 3,
    "pipe_radius": 0.5,
    "max_level": 0.9,
    "min_level": 0.1,
}

TANK_DIST = {"add": True, "nom_flow": 1, "var_flow": 0}  # 2.75
TANK_DIST["max_flow"] = TANK_DIST["nom_flow"] + TANK_DIST["var_flow"] * 3
TANK_DIST["min_flow"] = TANK_DIST["nom_flow"] - TANK_DIST["var_flow"] * 3
