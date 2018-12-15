from models.tank_model.tank import Tank
from models.tank_model.disturbance import InflowDist
from models.ANN_model import ANN_model
from visualize.window import Window
import numpy as np 
import matplotlib.pyplot as plt 
import random
import pygame
import sys
from drawnow import drawnow
#=============== PARAMETERS ==================#

# Model parameters
TANK_HEIGHT=10
TANK_RADIUS=2
TBCC = 10 # Time before choke change
DELAY=0

# Disturbance params
ADD_INFLOW = True
DIST_PIPE_RADIUS=1
DIST_DISTRIBUTION="gauss"
DIST_NOM_FLOW=1000
DIST_VARIANCE_FLOW=100


# Training parameters
MAX_TIME = 1000
OBSERVATIONS = 3
VALVE_POSITIONS= 2

# Render parameters
WINDOW_HEIGHT=350
WINDOW_WIDTH=500
RENDER=True
LIVE_REWARD_PLOT= False

EPISODES = 10

# ============ MAIN ===========================#
def main():
    #============= Initialize variables ===========#
    tank = Tank(TANK_HEIGHT,TANK_RADIUS) # get model
    if ADD_INFLOW:
        inflow_dist = InflowDist(DIST_PIPE_RADIUS,DIST_NOM_FLOW,DIST_VARIANCE_FLOW)
    rewards = [] 
    model = ANN_model(
        input_size=OBSERVATIONS,
        hidden_size=[10,10],
        output_size=VALVE_POSITIONS,
        max_level=tank.h
        ) # initialize prediction model

    # ============== RENDER========== #
    if RENDER:
        window = Window(tank,WINDOW_WIDTH,WINDOW_HEIGHT)
    
    # =============== Live plotting of rewards===========#
    if LIVE_REWARD_PLOT:
        plt.ion()  # enable interactivity
        fig = plt.figure(num="Max reward: {}".format(MAX_TIME))  # make a figure
        def plot_rewards():
            plt.plot(rewards,label="Episode number: {}".format(e))
            plt.legend()
    
    
    # ================= Running episodes =================#
    running=True
    for e in range(EPISODES):
        tank.reset() # Reset level in tank
        inflow_dist.reset() # resets inflow to norm flow
        level_history = [] 
        valve_history = []
        
        # Running state in the episode
        
        for t in range(MAX_TIME):
            if len(level_history) >= OBSERVATIONS:
                z_input = model.predict(level_history[-OBSERVATIONS:])
                dl = tank.get_dl_outflow(z=z_input)
                tank.change_level(dl)
            else:
                z_input=0.5

            # Add disturbance to tank
            if ADD_INFLOW:
                q_inn = inflow_dist.get_flow()
                if q_inn < 0:
                    q_inn=0
                dl_dist = tank.get_dl_inflow(q_inn)
                tank.change_level(dl_dist)    
            # Check terminate state
            if tank.l < tank.min or tank.l > tank.max:
                break
            
            # render tank
            if RENDER:
                running = window.Draw(z_input)
                if not running:
                    break

            valve_history.append(z_input)
            level_history.append(tank.l)
        rewards.append(t)
        
        # Live plot rewards
        if LIVE_REWARD_PLOT:
            if not running: 
                break
            drawnow(plot_rewards)

        
    pygame.display.quit()
    print("\nMean rewards for episodes: ", np.mean(rewards)) 
    print("Rewards for the last episode: ", rewards[-1])

if __name__ == "__main__":
    main()