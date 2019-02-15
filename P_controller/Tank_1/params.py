MAIN_PARAMS = {
'Episodes':1,
'Mean_episodes':20,
'Max_time':200,
'RENDER':True,
'LIVE_REWARD_PLOT':False 
}

# Agent parameters
AGENT_PARAMS = {
'SS_POSITION' :0.5, # steady state set position
'VALVE_START_POSITION':0.5
}

TANK_PARAMS = {
    'height':10,
    'init_level':0.5,
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


