MAIN_PARAMS = {
'EPISODES':2000,
'MEAN_EPISODE':20,
'MAX_TIME':200,
'RENDER':True,
'LIVE_REWARD_PLOT':False,
}

AGENT_PARAMS = {
'SS_POSITION' :0.5, 
'VALVE_START_POSITION':0.5,
'ACTION_DELAY':5,
'INIT_POSITION':0.5,
'EPSILON_MIN' : 0.01,
'EPSILON_DECAY' : 0.9,
'LEARNING_RATE' : 0.0001,
'HIDDEN_LAYER_SIZE': [5,5],
'BATCH_SIZE':1,
'MEMORY_LENGTH':1000,
'OBSERVATIONS':2, 
'VALVE_POSITIONS':10, 
'GAMMA' :0.98,    
'EPSILON' :1.0 ,
'SAVE_MODEL':True,
'LOAD_MODEL': False,
'TRAIN_MODEL':True 
}
AGENT_PARAMS['BUFFER_THRESH'] = AGENT_PARAMS['BATCH_SIZE']*4

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
    'nom_flow':2.75, 
    'var_flow':0, 
    'max_flow':5,
    'min_flow':0,
}
