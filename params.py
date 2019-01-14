
#=============== PARAMETERS ==================#

# Model parameters
TANK_HEIGHT=10
TANK_RADIUS=2
TBCC = 500 # Time before choke change
DELAY=0



# Disturbance params
ADD_INFLOW = True
DIST_PIPE_RADIUS=1
DIST_DISTRIBUTION="gauss"
DIST_NOM_FLOW=70
DIST_VARIANCE_FLOW=20
DIST_MAX_FLOW = 120


# Agent parameters
MAX_TIME = 10000
SS_POSITION = 0.5*TANK_HEIGHT # steady state set position
VALVE_START_POSITION=0.5
OBSERVATIONS = 2 # Number of time steos observed
VALVE_POSITIONS= 10 # Number of valve positions #TODO change name
GAMMA = 0.95    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001


# Render parameters
RENDER=False
LIVE_REWARD_PLOT= True

EPISODES = 100