from Tank_params import (
    TANK1_PARAMS,
    TANK2_PARAMS,
    TANK3_PARAMS,
    TANK4_PARAMS,
    TANK5_PARAMS,
    TANK6_PARAMS,
)
from Tank_params import (
    TANK1_DIST,
    TANK2_DIST,
    TANK3_DIST,
    TANK4_DIST,
    TANK5_DIST,
    TANK6_DIST,
)

MAIN_PARAMS = {
    "EPISODES": 30000,
    "MEAN_EPISODE": 10,
    "MAX_TIME": 200,
    "RENDER": False,
    "LIVE_REWARD_PLOT": False,
}

AGENT_PARAMS = {
    "N_TANKS": 6,
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.5,
    "ACTION_DELAY": [5, 4, 5, 4, 5, 4],
    "INIT_ACTION": 0,
    "EPSILON_MIN": 0.05,
    "EPSILON_DECAY": [0.995, 0.996, 0.997, 0.998, 0.998, 0.999],
    "LEARNING_RATE": 0.001,
    "HIDDEN_LAYER_SIZE": [10],
    "BATCH_SIZE": 10,
    "MEMORY_LENGTH": 10000,
    "OBSERVATIONS": 4,  # level, gradient, is_above 0.5, prevous valve position
    "VALVE_POSITIONS": 5,
    "GAMMA": 0.9,
    "EPSILON": 1,
    "SAVE_MODEL": True,
    "LOAD_MODEL": False,
    "TRAIN_MODEL": True,
    "MODEL_NAME": "",
}
AGENT_PARAMS["BUFFER_THRESH"] = AGENT_PARAMS["BATCH_SIZE"] * 1

# Model parameters Tank 1

TANK_PARAMS = [
    TANK1_PARAMS,
    TANK2_PARAMS,
    TANK3_PARAMS,
    TANK4_PARAMS,
    TANK5_PARAMS,
    TANK6_PARAMS,
]
TANK_DIST = [
    TANK1_DIST,
    TANK2_DIST,
    TANK3_DIST,
    TANK4_DIST,
    TANK5_DIST,
    TANK6_DIST,
]

for DIST in TANK_DIST:
    DIST["step_time"] = int(MAIN_PARAMS["MAX_TIME"] / 2)
    DIST["max_time"] = MAIN_PARAMS["MAX_TIME"]

for i in range(1, AGENT_PARAMS["N_TANKS"]):
    TANK_DIST[i]["add"] = False
