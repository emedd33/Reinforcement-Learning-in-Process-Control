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
    for e in range(EPISODES):
        environment.episode = e
        state,reward,action_delay, next_state = environment.reset() # Reset level in tank
        grads = []
        # Running through states in the episode
        for _ in range(MAX_TIME):    
            probs,z,action = agent.act(state[-1])
            if _ == 0:
                saved_probs = probs
            terminated, next_state = environment.get_next_state(z,state[-1]) 
            reward.append(environment.get_reward(next_state,terminated))
            state.append(next_state)
            dsoftmax = agent.softmax_grad(probs)[z,:]
            dlog = dsoftmax / probs[0,z]
            # state[-1] = state[-1].reshape(2,1)
            grad = state[-1].dot(dlog[None,:])
            grads.append(grad)
            
            if terminated:
                break 
            if environment.show_rendering:
                environment.render(z)
        for i in range(len(grads)):
            agent.model_w += LEARNING_RATE * grads[i] * sum([ r * (agent.gamma ** r) for t,r in enumerate(reward[i:])])
        # agent.remember(state[-1],z,next_state,reward,terminated)
        episode_reward.append(np.sum(reward))
        
        agent.status(episode_reward,e,round(saved_probs[0][0],3))
        
        # if (agent.is_ready(batch_size)):
        agent.b.append(np.mean(reward))
        #     agent.GP_replay(batch_size,np.mean(agent.b))
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
    # print("Max rewards for all episodes: {}".format(np.max(all_rewards))) 
    # print("Mean rewards for the last 10 episodes: {}".format(np.mean(all_rewards[-10:]))) 
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
