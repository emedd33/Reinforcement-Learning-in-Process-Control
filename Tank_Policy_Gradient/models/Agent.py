from collections import deque
from tensorflow import keras
from keras import backend as K
from params import NUMBER_OF_HIDDEN_LAYERS,OBSERVATIONS,VALVE_POSITIONS,MEMORY_LENGTH,\
    EPSILON,EPSILON_MIN,EPSILON_DECAY,LOAD_ANN_MODEL,GAMMA,LEARNING_RATE,TRAIN_MODEL,\
        BATCH_SIZE,MEAN_EPISODE,EPISODES,SAVE_ANN_MODEL
import numpy as np 
import random
import itertools
import math
class Agent():
    def __init__(self,
            hl_size=NUMBER_OF_HIDDEN_LAYERS,
            state_size=2,
        ): 
        self.state_size = state_size
        self.memory = [[],[],[],[],[]]
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON if TRAIN_MODEL else EPSILON_MIN # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE


    def remember(self, state,z,reward,terminated,adv):
        if TRAIN_MODEL:
            reward = self.discount_and_normalize_rewards(reward)
            data = np.array([state[:-1],z,state[1:],reward,terminated])
            for i,history in enumerate(data):
                self.memory[i].extend(history)
        norm = adv[0]+len(reward)
        mean_reward = np.mean(reward)
       
        adv[1] = (adv[1]/norm)*adv[0]+(mean_reward/norm)*len(reward)
        adv[0] += len(reward)
        return adv
                
        
         
    def act(self, states):
        z = np.random.uniform(0,1)
        return z
        
    def is_ready(self,batch_size):
        if not TRAIN_MODEL:
            return False
        if len(self.memory)< batch_size:
            return False
        return True

    def discount_and_normalize_rewards(self,episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative
        
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
        return discounted_episode_rewards


    def GP_replay(self):
        pass
        
    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
###############################################################

    def status(self,episode_reward,e):
        if e % MEAN_EPISODE == 0 and e != 0:
            mean_reward = np.mean(episode_reward[-10:])
            print("Mean rewards {}/{} episodes : {} ".format(e,EPISODES,mean_reward))
          