from params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST, TANK_PARAMS

MAIN_PARAMS["RENDER"] = True

AGENT_PARAMS["EPSILON"] = [0, 0]
AGENT_PARAMS["Z_VARIANCE"] = [0.01, 0.01]
AGENT_PARAMS["SAVE_MODEL"] = [False, False]
AGENT_PARAMS["LOAD_MODEL"] = [True, True]
AGENT_PARAMS["TRAIN_MODEL"] = [False, False]
AGENT_PARAMS["LOAD_ACTOR_NAME"] = [
    "Actor_Network_[5, 5]HL",
    "Actor_Network_[5, 5]HL",
]
AGENT_PARAMS["LOAD_CRITIC_NAME"] = [
    "Critic_Network_[5, 5]HL",
    "Critic_Network_[5, 5]HL",
]

TANK_DIST[0]["pre_def_dist"] = True
for i in range(AGENT_PARAMS["N_TANKS"]):
    TANK_PARAMS[i]["max_level"] = 0.75
    TANK_PARAMS[i]["min_level"] = 0.25

for i in range(1, AGENT_PARAMS["N_TANKS"]):
    TANK_DIST[i]["add"] = True
    TANK_DIST[i]["add_step"] = False
    TANK_DIST[i]["pre_def_dist"] = False
    TANK_DIST[i]["nom_flow"] = 0
    TANK_DIST[i]["var_flow"] = 0
    TANK_DIST[i]["min_flow"] = 0
