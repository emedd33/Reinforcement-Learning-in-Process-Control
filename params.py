
#=============== PARAMETERS ==================#
EPISODES = 200
MAX_TIME = 100
SAVE_ANN_MODEL=False

# Model parameters
TANK_HEIGHT=10
TANK_RADIUS=3
TANK_PIPE_RADIUS=0.378 #m
INIT_LEVEL=0.5 # initial water level for each episode

# Choke parameters
TBCC = 5 # Time before choke change
HARD_MAX=0.6
SOFT_MAX=0.55
HARD_MIN = 0.4
SOFT_MIN = 0.45

# Disturbance params
ADD_INFLOW = True
DIST_DISTRIBUTION="gauss"
DIST_NOM_FLOW=1 # m^3/s
DIST_VARIANCE_FLOW= 0.05 # m^3/s
DIST_MAX_FLOW = 0.65 #DIST_NOM_FLOW + 3*DIST_VARIANCE_FLOW # 
DIST_MIN_FLOW=  0 #DIST_NOM_FLOW - 2*DIST_VARIANCE_FLOW


# Agent parameters
SS_POSITION = 0.5*TANK_HEIGHT # steady state set position
VALVE_START_POSITION=0.5
OBSERVATIONS = 2 # Number of time steos observed
VALVE_POSITIONS= 2 # Number of valve positions 
GAMMA = 0.95    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.01
NUMBER_OF_HIDDEN_LAYERS = []
BATCH_SIZE=20


# Render parameters
RENDER=True
LIVE_REWARD_PLOT= True
