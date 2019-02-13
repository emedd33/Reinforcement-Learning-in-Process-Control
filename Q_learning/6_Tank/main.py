print("#### IMPORTING ####")

from models.environment import Environment
from models.Agent import Agent
from params import BATCH_SIZE,EPISODES,MAX_TIME,TBCC,MEAN_EPISODE,\
    LIVE_REWARD_PLOT,SAVE_ANN_MODEL,RENDER,NUMBER_OF_HIDDEN_LAYERS,N_TANKS,MAX_OBTAINED_REWARD
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np 
import keyboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    #============= Initialize variables ===========#
    max_mean_reward = MAX_OBTAINED_REWARD
    environment = Environment()
    agent = Agent()
    # ================= Running episodes =================#
    all_rewards = [] 
    all_mean_rewards = []
    batch_size = BATCH_SIZE
    for e in range(EPISODES):
        state,action_delay, next_state,episode_reward = environment.reset() # Reset level in tank
        # Running through states in the episode
        for _ in range(MAX_TIME):    
            if action_delay >= TBCC:
                action = agent.act(state)
                z = agent.action_choices[action]
                action_delay=0
            else:
                action_delay+=1
            terminated, next_state = environment.get_next_state(z,state) 
            reward = environment.get_reward(next_state,terminated)
            episode_reward.append(np.sum(reward))
            if action_delay >=TBCC or terminated:
                agent.remember(state,action,next_state,reward,terminated)
            state=next_state
            if True in terminated:
                break 
            if environment.show_rendering:
                environment.render(z)
        if e % MEAN_EPISODE == 0 and e != 0:
            mean_reward = np.mean(all_rewards[-MEAN_EPISODE:])
            all_mean_rewards.append(mean_reward)
            print("Mean rewards for the last {} of {}/{} episodes : {} explore T1: {} explore T6: {}".format(MEAN_EPISODE,e,EPISODES,mean_reward,round(agent.epsilon[0],2),round(agent.epsilon[5],2)))
            if SAVE_ANN_MODEL:
                if mean_reward > max_mean_reward:
                    for i,model in enumerate(agent.ANN_models):
                        model_name = "/ANN_"+ str(NUMBER_OF_HIDDEN_LAYERS)+"HL_" + str(i) 
                        model_path = "Tank_Q_learning_6/models/saved_models" + model_name+ ".h5"
                        model.save(model_path)
                    print("ANN_Model was saved")
                    max_mean_reward = mean_reward
        if (agent.is_ready(batch_size)):
            loss = agent.Qreplay(batch_size)
        if keyboard.is_pressed('ctrl+x'):
            break
        
        # Live plot rewards
        all_rewards.append(np.sum(np.array(episode_reward)))
      
        
        if keyboard.is_pressed('ctrl+x'):
                break
        if LIVE_REWARD_PLOT:
            environment.plot(all_rewards,agent.epsilon)       
        if not environment.running:
            break
                    
    print("##### {} EPISODES DONE #####".format(e))
    print("Max rewards for all episodes: {}".format(np.max(all_rewards))) 
    print("Mean rewards for the last 10 episodes: {}".format(np.mean(all_rewards[-10:]))) 
    plt.ioff()
    plt.clf()
    plt.plot(all_mean_rewards)
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
