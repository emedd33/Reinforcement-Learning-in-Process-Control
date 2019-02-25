from params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST

MAIN_PARAMS["RENDER"] = False
MAIN_PARAMS["LIVE_REWARD_PLOT"] = False

AGENT_PARAMS["EPSILON"] = 0
AGENT_PARAMS["SAVE_MODEL"] = False
AGENT_PARAMS["LOAD_MODEL"] = True
AGENT_PARAMS["TRAIN_MODEL"] = False

TANK_DIST["var_flow"] = 0.1
TANK_DIST["add"] = True
TANK_DIST["max_flow"] = TANK_DIST["nom_flow"] + TANK_DIST["var_flow"] * 3
TANK_DIST["min_flow"] = TANK_DIST["nom_flow"] - TANK_DIST["var_flow"] * 3

