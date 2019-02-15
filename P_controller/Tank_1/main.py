print("#### IMPORTING ####")
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('C:/Users/eskil/Google Drive/Skolearbeid/5. klasse/Master')

from P_controller.Tank_1.models.environment import Environment
from P_controller.Tank_1.models.p_controller import P_controller
from P_controller.Tank_1.params import MAIN_PARAMS,TANK_DIST,TANK_PARAMS,AGENT_PARAMS
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import numpy as np 
import keyboard

def main(kc=0.5): 
    environment = Environment(TANK_PARAMS,TANK_DIST,MAIN_PARAMS)
    controller = P_controller(environment,AGENT_PARAMS)
    h = [5]
    z =[]
    d =[]
    reward = []
    for _ in range(MAIN_PARAMS['Max_time']):
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
    print(np.sum(reward))
    _, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
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
    print("  Max time in each episode: {}".format(MAIN_PARAMS['Max_time']))
    reward = main()
