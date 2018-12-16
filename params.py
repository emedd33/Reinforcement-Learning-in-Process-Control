
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
DIST_NOM_FLOW=500
DIST_VARIANCE_FLOW=100


# Training parameters
MAX_TIME = 10000
OBSERVATIONS = 10 # input = state[i+1]- state[i]
VALVE_POSITIONS= 10

# Render parameters
RENDER=True
LIVE_REWARD_PLOT= True

EPISODES = 1000