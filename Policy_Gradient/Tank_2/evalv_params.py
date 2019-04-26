from params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST

MAIN_PARAMS["RENDER"] = True
MAIN_PARAMS["LIVE_REWARD_PLOT"] = False

AGENT_PARAMS["EPSILON"] = [0, 0]
AGENT_PARAMS["SAVE_MODEL"] = [False, False]
AGENT_PARAMS["LOAD_MODEL"] = [True, True]
AGENT_PARAMS["TRAIN_MODEL"] = [False, False]
AGENT_PARAMS["Z_VARIANCE"] = [0.01, 0.01]

for i in range(1, AGENT_PARAMS["N_TANKS"]):
    TANK_DIST[i]["add"] = True
    TANK_DIST[i]["add_step"] = False
    TANK_DIST[i]["pre_def_dist"] = False
    TANK_DIST[i]["nom_flow"] = 0
    TANK_DIST[i]["var_flow"] = 0
    TANK_DIST[i]["min_flow"] = 0

TANK_DIST[0]["pre_def_dist"] = True
