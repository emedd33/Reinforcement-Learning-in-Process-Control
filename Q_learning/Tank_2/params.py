
#=============== PARAMETERS ==================#

EPISODES = 2000
MEAN_EPISODE=10
MAX_TIME = 100
MAX_OBTAINED_REWARD = 176.4
LOAD_ANN_MODEL = False
SAVE_ANN_MODEL=False
TRAIN_MODEL = True
N_TANKS = 2

INIT_LEVEL=0.5 # initial water level for each episode

# Choke parameters
TBCC = 1 # Time before choke change

# Agent parameters
MEMORY_LENGTH = 3000
SS_POSITION = 0.5 # steady state set position
VALVE_START_POSITION=0
OBSERVATIONS = 3 # Last timestep + gradient of water level + choke position of prev tank
VALVE_POSITIONS= 10 # Number of valve positions 
GAMMA = 0.1    # discount rate
EPSILON = 1.0  # exploration rate

EPSILON_MIN = 0.001
EPSILON_DECAY = 0.95
LEARNING_RATE = 0.01 
NUMBER_OF_HIDDEN_LAYERS = [10]
BATCH_SIZE=40


# Render parameters
RENDER=True
LIVE_REWARD_PLOT= False

# Model parameters Tank 1
TANK1_PARAMS = {
    'height':10,
    'width':3,
    'pipe_radius':0.5,
    'max_level':0.9,
    'min_level':0.1
}

TANK1_DIST = {
    'add':True,
    'nom_flow':0.5,
    'var_flow':0.01,
    'max_flow':0.7,
    'min_flow':0.3,
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
    'var_flow':0.01,
    'max_flow':0.7,
    'min_flow':0.3,
}