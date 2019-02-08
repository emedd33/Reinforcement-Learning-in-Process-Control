from collections import deque
from tensorflow import keras
from keras import backend as K
from params import NUMBER_OF_HIDDEN_LAYERS,OBSERVATIONS,VALVE_POSITIONS,MEMORY_LENGTH,\
    EPSILON,EPSILON_MIN,EPSILON_DECAY,LOAD_ANN_MODEL,GAMMA,LEARNING_RATE,TRAIN_MODEL,\
        BATCH_SIZE,MEAN_EPISODE,EPISODES,SAVE_ANN_MODEL,DECAY_RATE
import numpy as np 
import random
import itertools

class Agent():
    def __init__(self,
            hl_size=NUMBER_OF_HIDDEN_LAYERS,
            state_size=1,
        ): 
        self.state_size = state_size
        self.memory = [[],[],[],[],[]]
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON if TRAIN_MODEL else EPSILON_MIN # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.decay_rate = DECAY_RATE

        self.model = self._build_model(hl_size)
        self.h,self.x,self.dlogps,self.drs = [],[],[],[]
    def _build_model(self,hl_size):
        model = {}
        model['W1'] =  np.random.randn(hl_size[0],self.state_size) / np.sqrt(self.state_size)
        model['W2'] = np.random.randn(hl_size[0]) / np.sqrt(hl_size[0])
        
        self.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } 
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }
        return model
    
    def policy_forward(self,x):
        x = x.reshape(1,)
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0 
        logp = np.dot(self.model['W2'], h)    
        p = self.sigmoid(logp)
        return p, h 

    def sigmoid(self,x): 
        return 1.0 / (1.0 + np.exp(-x)) 
    
    def policy_backward(self,eph, epdlogp,epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        #recursively compute error derivatives for both layers, this is the chain rule
        #epdlopgp modulates the gradient with advantage
        #compute updated derivative with respect to weight 2. It's the parameter hidden states transpose * gradient w/ advantage (then flatten with ravel())
        dW2 = np.dot(eph.T, epdlogp).ravel()
        #Compute derivative hidden. It's the outer product of gradient w/ advatange and weight matrix 2 of 2
        dh = np.outer(epdlogp, self.model['W2'])
        #apply activation
        dh[eph <= 0] = 0 # backpro prelu
        #compute derivative with respect to weight 1 using hidden states transpose and input observation
        dW1 = np.dot(dh.T, epx)
        #return both derivatives to update weights
        return {'W1':dW1, 'W2':dW2}

    def GP_replay(self,episode_number):
        epx = np.vstack(self.x) #obsveration
        eph = np.vstack(self.h) #hidden
        epdlogp = np.vstack(self.dlogps) #gradient
        epr = np.vstack(self.drs) #reward
        self.x,self.h,self.dlogps,self.drs = [],[],[],[] # reset array memory

        discounted_epr = self.discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = self.policy_backward(eph, epdlogp,epx)
        if episode_number % BATCH_SIZE == 0:
            for k,v in self.model.items():
                g = self.grad_buffer[k] # gradient
                self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            self.decay_exploration()    
        for k in self.model: self.grad_buffer[k] += grad[k]
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
                
        
         
    def act(self, x):
        aprob,h = self.policy_forward(x)
        z = np.random.normal(aprob,self.epsilon)
        z = 0 if z < 0 else z
        z = 1 if z > 1 else z
        self.x.append(x) # observation
        self.h.append(h) # hidden state
        self.dlogps.append(z - aprob)
        return aprob,z
        
    def is_ready(self,batch_size):
        if not TRAIN_MODEL:
            return False
        if len(self.memory)< batch_size:
            return False
        return True
    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        #initilize discount reward matrix as empty
        discounted_r = np.zeros_like(r)
        #to store reward sums
        running_add = 0
        #for each reward
        for t in reversed(range(0, r.size)):
            #if reward at index t is nonzero, reset the sum, since this was a game boundary (pong specific!)
            # if r[t] != 0: running_add = 0 
            #increment the sum 
            #https://github.com/hunkim/ReinforcementZeroToAll/issues/1
            running_add = running_add * self.gamma + r[t]
            #earlier rewards given more value over time 
            #assign the calculated sum to our discounted reward matrix
            discounted_r[t] = running_add
        return discounted_r.astype(float)


    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
###############################################################

    def status(self,episode_reward,e):
        if e % MEAN_EPISODE == 0 and e != 0:
            mean_reward = np.mean(episode_reward[-10:])
            print("Mean rewards {}/{} episodes : {} exploration: {}".format(e,EPISODES,mean_reward,round(self.epsilon,2)))
          