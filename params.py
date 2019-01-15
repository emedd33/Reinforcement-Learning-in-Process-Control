
#=============== PARAMETERS ==================#
EPISODES = 100
MAX_TIME = 500
SAVE_ANN_MODEL=False

# Model parameters
TANK_HEIGHT=10
TANK_RADIUS=3
TBCC = 200 # Time before choke change
DELAY=0
MAX_LEVEL=0.99
MIN_LEVEL=0.01
INIT_LEVEL=0.8

# Disturbance params
ADD_INFLOW = False
DIST_PIPE_RADIUS=1
DIST_DISTRIBUTION="gauss"
DIST_NOM_FLOW=500
DIST_VARIANCE_FLOW=50
DIST_MAX_FLOW = 1000
DIST_MIN_FLOW=300


# Agent parameters
SS_POSITION = 0.5*TANK_HEIGHT # steady state set position
VALVE_START_POSITION=0.5
OBSERVATIONS = 1 # Number of time steos observed
VALVE_POSITIONS= 3 # Number of valve positions #TODO change name
GAMMA = 0.95    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE=10


# Render parameters
RENDER=False
LIVE_REWARD_PLOT= True
