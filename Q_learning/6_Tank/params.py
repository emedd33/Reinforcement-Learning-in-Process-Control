
#=============== PARAMETERS ==================#

EPISODES = 3000
MEAN_EPISODE=10
MAX_TIME = 1000
MAX_OBTAINED_REWARD = 550
LOAD_ANN_MODEL = False
LOAD_MODEL_NAME = "580"
SAVE_ANN_MODEL=False
TRAIN_MODEL=True
N_TANKS = 6

INIT_LEVEL=0.5 # initial water level for each episode

# Choke parameters
TBCC = 1 # Time before choke change

# Agent parameters
MEMORY_LENGTH = 1000
SS_POSITION = 0.5 # steady state set position
VALVE_START_POSITION=0
OBSERVATIONS = 3 # Last timestep + gradient of water level + choke position of prev tank
VALVE_POSITIONS= 10 # Number of valve positions 
GAMMA = 0.2    # discount rate
EPSILON = 1  # exploration rate

EPSILON_MIN = 0.0001
EPSILON_DECAY = 0.95
LEARNING_RATE = 0.01
NUMBER_OF_HIDDEN_LAYERS = [5,5]
BATCH_SIZE=50


# Render parameters
RENDER=True
LIVE_REWARD_PLOT= False
