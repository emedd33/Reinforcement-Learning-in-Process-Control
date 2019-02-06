from collections import deque
from tensorflow import keras
from params import NUMBER_OF_HIDDEN_LAYERS,OBSERVATIONS,VALVE_POSITIONS,MEMORY_LENGTH,\
    EPSILON,EPSILON_MIN,EPSILON_DECAY,LOAD_ANN_MODEL,GAMMA,LEARNING_RATE,TRAIN_MODEL,\
        BATCH_SIZE
import numpy as np 
import random
import itertools
class Agent():
    def __init__(self,
            hl_size=NUMBER_OF_HIDDEN_LAYERS,
            state_size=2,
        ): 
        self.state_size = state_size
        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON if TRAIN_MODEL else EPSILON_MIN # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.buffer_thres = BATCH_SIZE
        self.ANN_model = self._build_ANN(state_size,hl_size,learning_rate=0.01)
        
    def _build_ANN(self,state_size,hl_size,learning_rate):    
        if LOAD_ANN_MODEL:
            model_name = "/ANN_"+ str(NUMBER_OF_HIDDEN_LAYERS)+"HL" 
            model_path = "Tank_Q-learning/models/saved_models" + model_name+ ".h5"
            model = keras.models.load_model(model_path)
            return model
        # Defining network model
        model = keras.Sequential()
        try:
            model.add(keras.layers.Dense(hl_size[0],input_shape=(state_size,),activation='relu'))
            for i in hl_size:
                model.add(keras.layers.Dense(i,activation='relu'))
        except IndexError: # Zero hidden layer
            model.add(keras.layers.Dense(1,input_shape=(state_size,),activation='relu'))
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
        return model        

    def remember(self, state, action, next_state,reward,done):
        if TRAIN_MODEL:
            dis_reward = np.sum(self.discounted_reward(reward))
            next_state=np.array(next_state)
            next_state=next_state.reshape(1,2)
            state = state.reshape(1,2)
            self.memory.append((state, action, next_state, dis_reward,done))

    def act_greedy(self,state):
        pred = self.ANN_model.predict(state) 
        return pred[0]
         
    def act(self, states):
        if np.random.rand() <= self.epsilon: # Exploration 
            random_action = random.uniform(0,1)
            return random_action
        return self.act_greedy(states) # Exploitation

    def is_ready(self,batch_size):
        if not TRAIN_MODEL:
            return False
        if len(self.memory)< batch_size:
            return False
        return True

    def discounted_reward(self,reward):
        discounted_r = np.zeros_like(reward)
        running_add=0
        for t in reversed(range(0,len(reward))):
            if reward[t] !=0: 
                running_add = 0
            running_add = running_add*self.gamma + reward[t]
            discounted_r[t] = running_add
        return discounted_r

    def Qreplay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, next_state,reward,done in minibatch:
            target = np.array([reward*action])
            targets_f.append(target)
            states.append(state[0])
        targets_f = np.array(targets_f)
        targets_f = targets_f.reshape(batch_size,1)
        history = self.ANN_model.fit(np.array(states), targets_f, epochs=1, verbose=0)
        self.decay_exploration()
        self.memory.clear()
            
        
    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay