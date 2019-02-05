
#=============== PARAMETERS ==================#

EPISODES = 500
MEAN_EPISODE=10
MAX_TIME = 100
MAX_OBTAINED_REWARD = 519.9
LOAD_ANN_MODEL = True
SAVE_ANN_MODEL=False
N_TANKS = 6

INIT_LEVEL=0.5 # initial water level for each episode

# Choke parameters
TBCC = 1 # Time before choke change

# Agent parameters
MEMORY_LENGTH = 10000
SS_POSITION = 0.5 # steady state set position
VALVE_START_POSITION=0
OBSERVATIONS = 3 # Last timestep + gradient of water level + choke position of prev tank
VALVE_POSITIONS= 10 # Number of valve positions 
GAMMA = 0.1    # discount rate
EPSILON = 1.0  # exploration rate

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.01 
NUMBER_OF_HIDDEN_LAYERS = [10]
BATCH_SIZE=20


# Render parameters
RENDER=True
LIVE_REWARD_PLOT= False
