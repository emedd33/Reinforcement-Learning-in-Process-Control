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
    "RENDER": False,
    "MAX_MEAN_REWARD": 400,  # minimum reward before saving model
}

AGENT_PARAMS = {
    "N_TANKS": 6,
    "SS_POSITION": 0.5,
    "VALVE_START_POSITION": 0.2,
    "ACTION_DELAY": [5, 5, 5, 5, 5, 5],
    "INIT_ACTION": 0.3,
    "VALVEPOS_UNCERTAINTY": 0,
    "EPSILON_DECAY": [1, 1, 1, 1, 1, 1],
    "ACTOR_LEARNING_RATE": [0.001, 0.001, 0.01, 0.01, 0.01, 0.01],
    "CRITIC_LEARNING_RATE": [0.001, 0.001, 0.01, 0.01, 0.01, 0.01],
    "HIDDEN_LAYER_SIZE": [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
    "BATCH_SIZE": 1,
    "MEMORY_LENGTH": 200,
    "OBSERVATIONS": 4,  # level, gradient, is_above 0.5, prevous valve position
    "GAMMA": 0.99,
    "EPSILON": [0.01, 0.01, 0.05, 0.05, 0.05, 0.05],
    "EPSILON_MIN": [0.01, 0.01, 0.05, 0.05, 0.05, 0.05],
    "Z_VARIANCE": [0.01, 0.01, 0.05, 0.05, 0.05, 0.05],
    "SAVE_MODEL": [True, True, True, True, True, True],
    "LOAD_MODEL": [True, True, False, False, False, False],
    "TRAIN_MODEL": [False, False, True, True, True, True],
    "LOAD_ACTOR_NAME": [
        "Actor_Network_[5, 5]HL",
        "Actor_Network_[5, 5]HL",
        "Actor_Network_[5, 5]HL",
        "Actor_Network_[5, 5]HL",
        "Actor_Network_[5, 5]HL",
        "Actor_Network_[5, 5]HL",
    ],
    "LOAD_CRITIC_NAME": [
        "Critic_Network_[5, 5]HL",
        "Critic_Network_[5, 5]HL",
        "Critic_Network_[5, 5]HL",
        "Critic_Network_[5, 5]HL",
        "Critic_Network_[5, 5]HL",
        "Critic_Network_[5, 5]HL",
    ],
    "LOAD_MODEL_PATH": "Actor_Critic/Tank_6/saved_networks/usable_networks/",
    "SAVE_MODEL_PATH": "Actor_Critic/Tank_6/saved_networks/training_networks/",
}

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

from rewards import get_reward_3 as get_reward
from rewards import sum_rewards
