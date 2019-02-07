from collections import deque
from tensorflow import keras
from keras import backend as K
from params import NUMBER_OF_HIDDEN_LAYERS,OBSERVATIONS,VALVE_POSITIONS,MEMORY_LENGTH,\
    EPSILON,EPSILON_MIN,EPSILON_DECAY,LOAD_ANN_MODEL,GAMMA,LEARNING_RATE,TRAIN_MODEL,\
        BATCH_SIZE,MEAN_EPISODE,EPISODES,SAVE_ANN_MODEL
import numpy as np 
import random
import itertools
class Agent():
    def __init__(self,
            hl_size=NUMBER_OF_HIDDEN_LAYERS,
            state_size=2,
        ): 
        self.b = [-0.48]
        self.state_size = state_size
        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON if TRAIN_MODEL else EPSILON_MIN # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.buffer_thres = BATCH_SIZE
        
        self.model_w = np.random.rand(1,2)

    def policy(self,state):
        # state = state.reshape(1,2)
        z = state.dot(self.model_w)
        exp = np.exp(z)
        return exp/np.sum(exp)

    # Vectorized softmax Jacobian
    def softmax_grad(self,softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def remember(self, state, action, next_state,reward,done):
        if TRAIN_MODEL:
            dis_reward = self.discount_and_normalize_rewards(reward)
            next_state=np.array(next_state)
            next_state=next_state.reshape(1,2)
            state = state.reshape(1,2)
            self.memory.append((state, action, next_state, reward[-1],done))

    def act_greedy(self,state):
        pred = self.ANN_model.predict(state) 
        action =(np.random.normal(pred[0],0.01))
        action = 0.8 if action > 0.8 else action
        action = 0.2 if action < 0.2 else action
        return action
         
    def act(self, states):
        # if np.random.rand() <= self.epsilon: # Exploration 
        #     random_action = random.uniform(0,1)
        #     return random_action
        probs = self.policy(states)
        z = np.random.choice(2,p=probs[0])
        action = np.array([0,0])
        action[int(z)] = 1
        return probs,z,action

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


    def GP_replay(self,batch_size,b):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f, actions = [], [], []
        for state, action, next_state,reward,done in minibatch:
            target = reward
            targets_f.append(target)
            states.append(state[0])
            actions.append(action)
        targets_f = np.array(targets_f)
        targets_f = targets_f.reshape(batch_size,1)
        states = np.array(states)
        actions = np.array(actions)
        self.PG_ANN_model.train_on_batch([states, targets_f], actions)
        # history = self.ANN_model.fit(np.array(states), targets_f, epochs=1, verbose=0)
        self.decay_exploration()
            
        
    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
###############################################################

    def status(self,episode_reward,e,prob_0):
        if e % MEAN_EPISODE == 0 and e != 0:
            mean_reward = np.mean(episode_reward[-10:])
            print("Mean rewards {}/{} episodes : {} b: {} start prob going down : {}".format(e,EPISODES,mean_reward,round(np.mean(self.b),2),prob_0))
            if SAVE_ANN_MODEL:
                max_mean_reward = 10000
                if mean_reward > max_mean_reward:
                    for i,model in enumerate(self.ANN_models):
                        model_name = "/ANN_"+ str(NUMBER_OF_HIDDEN_LAYERS)+"HL_" + str(i) 
                        model_path = "Tank_Q_learning_2/models/saved_models" + model_name+ ".h5"
                        model.save(model_path)
                    print("ANN_Model was saved")
                    max_mean_reward = mean_reward