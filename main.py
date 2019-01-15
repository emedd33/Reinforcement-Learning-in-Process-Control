from models.environment import Environment
from models.Agent import Agent
from params import * # Parameters used in main
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
    running=True
    all_rewards = [] 
    batch_size = BATCH_SIZE
    reward_index = int(0)
    for e in range(EPISODES):
        state, action, next_state,episode_reward = environment.reset() # Reset level in tank
        # Running through states in the episode
        for t in range(MAX_TIME):    
            action = agent.act(state)
            z = agent.action_choices[action]
            terminated, next_state = environment.get_next_state(z,state) 
            reward = environment.get_reward(next_state,terminated,t)
            agent.remember(state,action,next_state,reward,terminated)
            episode_reward+=reward
            if terminated:
                break 
            if environment.show_rendering:
                environment.render(z,next_state[-1])
            if len(agent.memory)>batch_size:
                agent.replay(batch_size)
        # Live plot rewards
        agent.decay_exploration()
        all_rewards.append(episode_reward)
        if LIVE_REWARD_PLOT:
            if keyboard.is_pressed('ctrl+c'):
                break
            else:
                environment.plot(all_rewards,agent.epsilon)       
        if not environment.running:
            break
                    
        
        
    print("##### {} EPISODES DONE #####".format(EPISODES))
    print("Mean rewards for all episodes: {}".format(np.mean(all_rewards))) 
    print("Mean rewards for the last 10 episodes: {}".format(np.mean(all_rewards[-10:]))) 
    if SAVE_ANN_MODEL:
        print("ANN_Model was saved")
if __name__ == "__main__":
    main()