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

MAIN_PARAMS = {"EPISODES": 1, "MAX_TIME": 200, "RENDER": True}
# Agent parameters
AGENT1_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 200,
}

AGENT2_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 250,
}
AGENT3_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 250,
}

AGENT4_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 250,
}

AGENT5_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 250,
}

AGENT6_PARAMS = {
    "SS_POSITION": 0.5,  # steady state set position
    "ACTION_DELAY": 5,
    "INIT_POSITION": 0.2,
    "TAU_C": 250,
}

AGENT_PARAMS_LIST = [
    AGENT1_PARAMS,
    AGENT2_PARAMS,
    AGENT3_PARAMS,
    AGENT4_PARAMS,
    AGENT5_PARAMS,
    AGENT6_PARAMS,
]
TANK_PARAMS_LIST = [
    TANK1_PARAMS,
    TANK2_PARAMS,
    TANK3_PARAMS,
    TANK4_PARAMS,
    TANK5_PARAMS,
    TANK6_PARAMS,
]
TANK_DIST_LIST = [
    TANK1_DIST,
    TANK2_DIST,
    TANK3_DIST,
    TANK4_DIST,
    TANK5_DIST,
    TANK6_DIST,
]

for DIST in TANK_DIST_LIST:
    DIST["step_time"] = int(MAIN_PARAMS["MAX_TIME"] / 2)
    DIST["max_time"] = MAIN_PARAMS["MAX_TIME"]

for i in range(1, 6):
    TANK_DIST_LIST[i]["add"] = False
