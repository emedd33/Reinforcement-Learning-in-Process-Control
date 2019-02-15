
EPISODES = 2000
MEAN_EPISODE=20
MAX_TIME = 200
SAVE_ANN_MODEL=False
LOAD_ANN_MODEL = False
TRAIN_MODEL=True
# Render parameters, will increase run time when set to True
RENDER=False
LIVE_REWARD_PLOT= False 

# Agent parameters
MEMORY_LENGTH=3000
SS_POSITION = 0.5 # steady state set position
VALVE_START_POSITION=0.5
OBSERVATIONS = 2 # Number of time steos observed
VALVE_POSITIONS= 10 # Number of valve positions 
GAMMA = 0.98    # discount rate
EPSILON = 1.0  # exploration rate

EPSILON_MIN = 0
EPSILON_DECAY = 0.95
LEARNING_RATE = 0.00001
NUMBER_OF_HIDDEN_LAYERS = [5,5]
BATCH_SIZE=100

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
    'var_flow':0.1,
    'max_flow':5,
    'min_flow':0,
}


