
#=============== PARAMETERS ==================#

EPISODES = 200
MEAN_EPISODE=10
MAX_TIME = 100
SAVE_ANN_MODEL=False
N_TANKS = 2
# Model parameters Tank 1
TANK1_PARAMS = {
    'height':10,
    'width':3,
    'pipe_radius':0.5,
    'max_level':0.9,
    'min_level':0.1
}

TANK1_DIST = {
    'add':False,
    'nom_flow':0.5,
    'var_flow':0.02,
    'max_flow':1,
    'min_flow':0,
}
# Model parameters Tank 1
TANK2_PARAMS = {
    'height':10,
    'width':3,
    'pipe_radius':0.5,
    'max_level':0.9,
    'min_level':0.1
}
TANK2_DIST = {
    'add':False,
    'nom_flow':0.5,
    'var_flow':0.02,
    'max_flow':1,
    'min_flow':0,
}

INIT_LEVEL=0.5 # initial water level for each episode

# Choke parameters
TBCC = 1 # Time before choke change

# Agent parameters
SS_POSITION = 0.5 # steady state set position
VALVE_START_POSITION=0
OBSERVATIONS = 2 # Number of time steos observed
VALVE_POSITIONS= 3 # Number of valve positions 
GAMMA = 0    # discount rate
EPSILON = 1.0  # exploration rate

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.1
NUMBER_OF_HIDDEN_LAYERS = [15,15]
BATCH_SIZE=10


# Render parameters
RENDER=True
LIVE_REWARD_PLOT= False
