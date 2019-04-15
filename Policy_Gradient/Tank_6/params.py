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
    "EPISODES": 50000,
    "MEAN_EPISODE": 5,
    "MAX_TIME": 200,
    "RENDER": True,
    "LIVE_REWARD_PLOT": False,
}

AGENT_PARAMS = {
    "N_TANKS": 6,
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.5,
    "ACTION_DELAY": [5, 5, 5, 5, 5, 5],
    "INIT_ACTION": 0,
    "EPSILON_MIN": 0.05,
    "VALVEPOS_UNCERTAINTY": 0.02,
    "EPSILON_DECAY": [0.995, 0.996, 0.997, 0.998, 0.998, 0.999],
    "LEARNING_RATE": 0.0005,
    "HIDDEN_LAYER_SIZE": [5, 5],
    "BATCH_SIZE": 5,
    "MEMORY_LENGTH": 10000,
    "OBSERVATIONS": 4,  # level, gradient, is_above 0.5, prevous valve position
    "GAMMA": 0.5,
    "EPSILON": 0.05,
    "Z_VARIANCE": 0.05,
    "SAVE_MODEL": True,
    "LOAD_MODEL": False,
    "TRAIN_MODEL": True,
    "MODEL_NAME": "Network_[5, 5]HL",
    "BASE_LINE_MEAN_REWARDS": 20,
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
