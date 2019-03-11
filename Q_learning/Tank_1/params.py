MAIN_PARAMS = {
    "EPISODES": 10000,
    "MEAN_EPISODE": 10,
    "MAX_TIME": 200,
    "RENDER": False,
    "LIVE_REWARD_PLOT": False,
}

AGENT_PARAMS = {
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.5,
    "ACTION_DELAY": 5,
    "INIT_ACTION": 0,
    "EPSILON_MIN": 0,
    "EPSILON_DECAY": 0.995,
    "LEARNING_RATE": 0.0005,
    "HIDDEN_LAYER_SIZE": [10],
    "BATCH_SIZE": 1,
    "MEMORY_LENGTH": 10000,
    "OBSERVATIONS": 3,
    "VALVE_POSITIONS": 10,
    "GAMMA": 0.5,
    "EPSILON": 1,
    "SAVE_MODEL": True,
    "LOAD_MODEL": False,
    "TRAIN_MODEL": True,
    "MODEL_NAME": "Network_[5]HL108",
}
AGENT_PARAMS["BUFFER_THRESH"] = AGENT_PARAMS["BATCH_SIZE"] * 1

TANK_PARAMS = {
    "height": 10,
    "init_level": 0.5,
    "width": 3,
    "pipe_radius": 0.5,
    "max_level": 0.75,
    "min_level": 0.25,
}

TANK_DIST = {
    "add": True,
    "nom_flow": 1,  # 2.7503
    "var_flow": 0.1,
    "max_flow": 2,
    "min_flow": 0.7,
    "add_step": False,
    "step_time": int(MAIN_PARAMS["MAX_TIME"] / 2),
    "step_flow": 2,
}
