from params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST, TANK1_PARAMS

MAIN_PARAMS["RENDER"] = True
MAIN_PARAMS["LIVE_REWARD_PLOT"] = False

AGENT_PARAMS["EPSILON"] = 0
AGENT_PARAMS["SAVE_MODEL"] = False
AGENT_PARAMS["LOAD_MODEL"] = True
AGENT_PARAMS["TRAIN_MODEL"] = False
AGENT_PARAMS["MODEL_NAME"] = "Network_[5, 5]HL"

TANK1_PARAMS["max_level"] = 0.75
TANK1_PARAMS["min_level"] = 0.25

for i in range(AGENT_PARAMS["N_TANKS"]):
    TANK_DIST[i]["add"] = True
    TANK_DIST[i]["var_flow"] = 0.1
    TANK_DIST[i]["add_step"] = True

