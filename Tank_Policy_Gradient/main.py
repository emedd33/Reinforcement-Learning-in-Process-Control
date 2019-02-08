print("#### IMPORTING ####")

from models.environment import Environment
from models.Agent import Agent
from params import BATCH_SIZE,EPISODES,MAX_TIME,TBCC,MEAN_EPISODE,\
    LIVE_REWARD_PLOT,SAVE_ANN_MODEL,RENDER,NUMBER_OF_HIDDEN_LAYERS,N_TANKS,LEARNING_RATE
import os
import matplotlib.pyplot as plt
import numpy as np 
import keyboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    #============= Initialize variables ===========#
    environment = Environment()
    agent = Agent()
    # ================= Running episodes =================#
    batch_size = BATCH_SIZE
    episode_reward = []
    adv,state,reward,actions,terminateds = [0,0],[],[],[],[]

    for e in range(EPISODES):
        environment.episode = e
        state.append(environment.reset()) # Reset level in tank
        grads = []
        # Running through states in the episode
        for _ in range(MAX_TIME):    
            actions.append(agent.act(state[-1]))
   
            terminated, next_state = environment.get_next_state(actions[-1],state[-1]) 
            reward.append(environment.get_reward(next_state,terminated))
            state.append(next_state)
            terminateds.append(terminated)
            if environment.show_rendering:
                environment.render(actions[-1])
            if terminated:
                break 
        adv = agent.remember(state,actions,reward,terminateds,adv)
        if len(agent.memory)>=batch_size:
            agent.GP_replay()

        episode_reward.append(np.sum(reward))
        state,reward,actions,terminateds = [],[],[],[]
        agent.status(episode_reward,e)
        if keyboard.is_pressed('ctrl+x'):
            break
        
        # Live plot rewards
        
        if keyboard.is_pressed('ctrl+x'):
                break
        if LIVE_REWARD_PLOT:
            environment.plot(all_rewards,agent.epsilon)       
        if not environment.running:
            break
                    
    print("##### {} EPISODES DONE #####".format(e+1))
    plt.ioff()
    plt.clf()
    plt.plot(episode_reward)
    plt.ylabel('Episodic reward')
    plt.xlabel('Episode')
    plt.show()
    
if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max number of episodes: {}".format(EPISODES))
    print("  Max time in each episode: {}".format(MAX_TIME))
    print("  Max reward in each episode: {}".format(MAX_TIME*N_TANKS))
    print("  {}Rendring simulation ".format("" if RENDER else "Not "))
    main()
