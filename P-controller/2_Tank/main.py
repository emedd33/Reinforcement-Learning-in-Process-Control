print("#### IMPORTING ####")

from models.environment import Environment
from models.p_controller import P_controller
from params import BATCH_SIZE,EPISODES,MAX_TIME,MEAN_EPISODE,\
    LIVE_REWARD_PLOT,SAVE_ANN_MODEL,RENDER,NUMBER_OF_HIDDEN_LAYERS,TANK_DIST
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np 
import keyboard

def main(kc=0.5):
    environment = Environment()
    controller = P_controller(environment)
    h = [5]
    z =[]
    d =[]
    episode_time=100
    reward = []
    for i in range(episode_time):
        new_z = controller.get_z(h[-1])
        z.append(new_z)
        new_h = environment.get_next_state(z[-1])
        new_reward = environment.get_reward(h[-1])
        reward.append(new_reward)
        if TANK_DIST['add']:
            new_d = environment.model.dist.flow
            d.append(new_d)
        h.append(new_h)
        if environment.show_rendering:
                environment.render(z[-1])
        if keyboard.is_pressed('ctrl+x'):
            break
    
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    l1,=ax1.plot(h,color="peru", ) 
    ax1.set_ylim(2,6)   
    l2,=ax2.plot(z,color="firebrick")
    ax2.set_ylim(0,1)
    l3,=ax3.plot(d, color="dimgray")

    
    plt.legend([l1, l2, l3],["Tank height", "Valve position", "Disturbance"], )
    plt.tight_layout()
    plt.xlabel("Time")
    plt.show()
    return np.sum(reward)
if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max time in each episode: {}".format(MAX_TIME))
    reward = main()
