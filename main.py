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

WINDOW_HEIGHT=400
WINDOW_WIDTH=300

RENDER=False
PLOT= False

EPISODES = 1
# ============ MAIN ===========================#
def main():
    tank = Tank(TANK_HEIGHT,TANK_RADIUS)
    time_history = range(MAX_TIME)
    scores = []
    # ============== RENDER========== #
    if RENDER:
        window = Window(tank)
    for _ in range(EPISODES):
        tank.reset()
        level_history = []
        valve_history = []
        done = False
        
        random_valve =  np.array([int(MAX_TIME/10)*[random.uniform(0,1)] for i in range(10)])
        valve_history = random_valve.reshape(-1)

        
        for t in range(MAX_TIME):
            tank.z = valve_history[t]
            if RENDER:
                running = window.Draw()
                if not running:
                    break

            level_history.append(tank.l)
            tank.change_level()
            if tank.l < tank.min or tank.l > tank.max:
                break
        scores.append(t)
        
        if PLOT:
            plt.plot(level_history)
            plt.plot(valve_history)
            plt.show()
    pygame.display.quit()
    print(np.mean(scores)) 
            
main()