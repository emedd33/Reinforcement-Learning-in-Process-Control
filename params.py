
#=============== PARAMETERS ==================#
EPISODES = 1000
MAX_TIME = 500
SAVE_ANN_MODEL=False

# Model parameters
TANK_HEIGHT=10
TANK_RADIUS=3
TANK_PIPE_RADIUS=0.5 #m
INIT_LEVEL=0.5 # initial water level for each episode

# Choke parameters
TBCC = 200 # Time before choke change
MAX_LEVEL=0.9
MIN_LEVEL=0.1

# Disturbance params
ADD_INFLOW = True
DIST_DISTRIBUTION="gauss"
DIST_PIPE_RADIUS=0.1778
DIST_NOM_FLOW=1 # m^3/s
DIST_VARIANCE_FLOW= 0.1 # m^3/s
DIST_MAX_FLOW = DIST_NOM_FLOW + 3*DIST_VARIANCE_FLOW # 
DIST_MIN_FLOW=  DIST_NOM_FLOW - 3*DIST_VARIANCE_FLOW


# Agent parameters
SS_POSITION = 0.5*TANK_HEIGHT # steady state set position
VALVE_START_POSITION=0.5
OBSERVATIONS = 1 # Number of time steos observed
VALVE_POSITIONS= 2 # Number of valve positions 
GAMMA = 0.95    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.001
NUMBER_OF_HIDDEN_LAYERS = [5,5]
BATCH_SIZE=20


# Render parameters
RENDER=True
LIVE_REWARD_PLOT= True
