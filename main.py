from tank_model.tank import Tank
from visualize.window import Window
import numpy as np 
import matplotlib.pyplot as plt 
import random
import pygame
import sys
#=============== PARAMETERS ==================#
TANK_HEIGHT=10
TANK_RADIUS=3
MAX_TIME = 10000

WINDOW_HEIGHT=500
WINDOW_WIDTH=500
TBCC = 100 # Time before choke change
RENDER=True
PLOT= True
DELAY=100
ADD_INFLOW = False

EPISODES = 1
# ============ MAIN ===========================#
def main():
    tank = Tank(TANK_HEIGHT,TANK_RADIUS)
    time_history = range(MAX_TIME)
    scores = []
    # ============== RENDER========== #
    if RENDER:
        window = Window(tank,WINDOW_HEIGHT,WINDOW_WIDTH)
    for _ in range(EPISODES):
        tank.reset()
        level_history = DELAY*[tank.l]
        valve_history = []
        done = False
        
        _ = 0
        while _ < MAX_TIME:
            valve_history.extend(TBCC*[np.random.uniform(0,1)])
            _+=TBCC

        valve_history = np.array(valve_history).reshape(-1)

        
        for t in range(MAX_TIME):
            input_z = valve_history[t]
            if RENDER:
                running = window.Draw(input_z)
                if not running:
                    break

            level_history.append(tank.l)

            tank.change_level(z=input_z)
            if tank.l < tank.min or tank.l > tank.max:
                break
            if ADD_INFLOW:
                pass
        scores.append(t)

        if PLOT:
            plt.plot(level_history)
            plt.plot(valve_history[:t+DELAY])
            plt.show()
    pygame.display.quit()
    print(np.mean(scores)) 
            
main()