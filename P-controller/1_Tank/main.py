print("#### IMPORTING ####")

from models.environment import Environment
from models.p_controller import P_controller
from params import BATCH_SIZE,EPISODES,MAX_TIME,MEAN_EPISODE,\
    LIVE_REWARD_PLOT,SAVE_ANN_MODEL,RENDER,NUMBER_OF_HIDDEN_LAYERS,TANK_DIST
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np 
import keyboard

def main():
    environment = Environment()
    controller = P_controller(environment)
    h = [0.2]
    z =[]
    d =[]
    for i in range(100000):
        new_z = controller.get_z(h[-1])
        z.append(new_z)
        new_h = environment.get_next_state(z[-1])
        if TANK_DIST['add']:
            new_d = environment.model.dist.flow
            d.append(new_d)
        h.append(new_h)
        if environment.show_rendering:
                environment.render(z[-1])
        if keyboard.is_pressed('ctrl+x'):
            break
    plt.subplot(3,1,1)  
    plt.plot(h,label="h")
    plt.subplot(3,1,2)
    plt.plot(z,label="z")
    plt.subplot(3,1,3)
    plt.plot(d,label="d")
    plt.show()
    

if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max time in each episode: {}".format(MAX_TIME))
    main()
