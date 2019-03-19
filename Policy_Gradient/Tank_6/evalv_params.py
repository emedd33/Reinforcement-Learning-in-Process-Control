from params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST

MAIN_PARAMS["RENDER"] = True
MAIN_PARAMS["LIVE_REWARD_PLOT"] = False

AGENT_PARAMS["EPSILON"] = 0
AGENT_PARAMS["SAVE_MODEL"] = False
AGENT_PARAMS["LOAD_MODEL"] = True
AGENT_PARAMS["TRAIN_MODEL"] = False
AGENT_PARAMS["MODEL_NAME"] = "Network_[10]HL"


for i in range(AGENT_PARAMS["N_TANKS"]):
    TANK_DIST[i]["add"] = True
    TANK_DIST[i]["var_flow"] = 0
    TANK_DIST[i]["add_step"] = True
    TANK_DIST[i]["pre_def_dist"] = True
