
#=============== PARAMETERS ==================#
EPISODES = 100
MAX_TIME = 20000
SAVE_ANN_MODEL=False

# Model parameters
TANK_HEIGHT=10
TANK_RADIUS=3
TBCC = 200 # Time before choke change
DELAY=0
MAX_LEVEL=0.99
MIN_LEVEL=0.01


# Disturbance params
ADD_INFLOW = True
DIST_PIPE_RADIUS=1
DIST_DISTRIBUTION="gauss"
DIST_NOM_FLOW=120
DIST_VARIANCE_FLOW=20
DIST_MAX_FLOW = 200


# Agent parameters
SS_POSITION = 0.5*TANK_HEIGHT # steady state set position
VALVE_START_POSITION=0.5
OBSERVATIONS = 2 # Number of time steos observed
VALVE_POSITIONS= 3 # Number of valve positions #TODO change name
GAMMA = 0.95    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE=50


# Render parameters
RENDER=False
LIVE_REWARD_PLOT= False
