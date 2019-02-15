
EPISODES = 1
MEAN_EPISODE=20
MAX_TIME = 200

# Render parameters, will increase run time when set to True
RENDER=False
LIVE_REWARD_PLOT= False 

# Agent parameters
SS_POSITION = 0.5 # steady state set position

# Model parameters Tank 1
INIT_LEVEL=0.5 # initial water level for each episode
TANK_PARAMS = {
    'height':10,
    'width':3,
    'pipe_radius':0.5,
    'max_level':0.9,
    'min_level':0.1
}

TANK_DIST = {
    'add':True,
    'nom_flow':2, # 2.7503
    'var_flow':0,
    'max_flow':5,
    'min_flow':0,
}


