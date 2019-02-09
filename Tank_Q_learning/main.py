print("#### IMPORTING ####")

from models.environment import Environment
from models.Agent import Agent
from params import BATCH_SIZE,EPISODES,MAX_TIME,MEAN_EPISODE,\
    LIVE_REWARD_PLOT,SAVE_ANN_MODEL,RENDER,NUMBER_OF_HIDDEN_LAYERS
import os
import matplotlib.pyplot as plt
import numpy as np 
import keyboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    #============= Initialize variables and objects ===========#
    max_mean_reward = 0
    environment = Environment()
    agent = Agent()
    
    all_rewards = [] 
    all_mean_rewards = []
    batch_size = BATCH_SIZE
    
    # ================= Running episodes =================#
    for e in range(EPISODES):
        state, next_state,episode_reward = environment.reset() # Reset level in tank
        
        for _ in range(MAX_TIME):    
            action = agent.act(state) # get action choice from state
            z = agent.action_choices[action] # convert action choice into valve position
            terminated, next_state = environment.get_next_state(z,state)  # Calculate next state with action
            reward = environment.get_reward(next_state,terminated) # get reward from transition to next state
            
            # Store data 
            episode_reward.append(np.sum(reward)) 
            agent.remember(state,action,next_state,reward,terminated)
            
            state=next_state
            
            if environment.show_rendering:
                environment.render(z)
            if terminated:
                break 
            # End for
        all_rewards.append(np.sum(np.array(episode_reward)))
        
        # Print mean reward and save better models
        if e % MEAN_EPISODE == 0 and e != 0:
            mean_reward = np.mean(all_rewards[-MEAN_EPISODE:])
            all_mean_rewards.append(mean_reward)
            print("Mean rewards for the last {} of {}/{} episodes : {} explore: {}".format(MEAN_EPISODE,e,EPISODES,mean_reward,round(agent.epsilon,2)))
            if SAVE_ANN_MODEL:
                max_mean_reward = agent.save_model(mean_reward,max_mean_reward)
        
        # Train model
        if (agent.is_ready(batch_size)):
            agent.Qreplay(batch_size)
        
        if keyboard.is_pressed('ctrl+c'):
            break
        
        if LIVE_REWARD_PLOT:
            environment.plot(all_rewards,agent.epsilon)       
        if not environment.running:
            break
                    
    print("##### {} EPISODES DONE #####".format(e+1))
    print("Max rewards for all episodes: {}".format(np.max(all_rewards))) 
    plt.ioff()
    plt.clf()
    plt.plot(all_mean_rewards)
    plt.ylabel('10 Episodic mean rewards')
    plt.xlabel('Episode')
    plt.show()
    
if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max number of episodes: {}".format(EPISODES))
    print("  Max time in each episode: {}".format(MAX_TIME))
    print("  {}Rendring simulation ".format("" if RENDER else "Not "))
    main()
