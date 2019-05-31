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
    "EPISODES": 20000,
    "MEAN_EPISODE": 50,
    "MAX_TIME": 200,
    "RENDER": True,
    "MAX_MEAN_REWARD": 300,  # minimum reward before saving model
}

AGENT_PARAMS = {
    "N_TANKS": 6,
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.2,
    "ACTION_DELAY": [5, 5, 5, 5, 5, 5],
    "INIT_ACTION": 0,
    "VALVE_POSITIONS": 10,
    "EPSILON": [1, 1, 1, 1, 1, 1],
    "EPSILON_MIN": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    "EPSILON_DECAY": [0.999, 0.9995, 0.999, 0.9995, 0.999, 0.9995],
    "LEARNING_RATE": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "HIDDEN_LAYER_SIZE": [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
    "BATCH_SIZE": 5,
    "MEMORY_LENGTH": 10000,
    "OBSERVATIONS": 4,  # level, gradient, is_above 0.5, prevous valve position
    "GAMMA": 0.9,
    "SAVE_MODEL": [True, True, True, True, True, True],
    "LOAD_MODEL": [False, False, False, False, False, False],
    "TRAIN_MODEL": [True, True, True, True, True, True],
    "LOAD_MODEL_NAME": ["", "", "", "", "", ""],
    "LOAD_MODEL_PATH": "Q_learning/Tank_6/",
    "SAVE_MODEL_PATH": "Q_learning/Tank_6/",
}

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
