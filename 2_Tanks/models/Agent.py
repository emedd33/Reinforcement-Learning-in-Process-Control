from collections import deque
from tensorflow import keras
from params import *
import numpy as np 
import random
import itertools
class Agent():
    def __init__(self,
            hl_size=NUMBER_OF_HIDDEN_LAYERS,
            state_size=[N_TANKS,OBSERVATIONS],
            action_size=VALVE_POSITIONS,
            n_tanks = N_TANKS
        ):

        self.state_size = state_size
        self.action_size = action_size
        self.action_choices = self._get_action_choices(n_tanks,action_size)
        self.memory = deque(maxlen=5000)
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON  # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.replay_counter = 0
        self.ANN_model = self._build_ANN(state_size,hl_size,action_size,learning_rate=0.01)
        
    
    def _get_action_choices(self,n_tanks,action_pos):
        # Create a list with all the valve positions
        all_valve_pos = []
        for j in range(n_tanks):
            valve_positions= []
            for i in range(action_pos):
                valve_positions.append((i)/(action_pos-1))
            all_valve_pos.append(np.array(list(reversed(valve_positions))))
        
        # Create all combinations of valve positions
        all_combinations = []
        for i in range(len(all_valve_pos)):
            arr = all_valve_pos[i]
            for element in arr:
                for j in range(i+1,len(all_valve_pos)):
                    for val in all_valve_pos[j]:
                        all_combinations.append([element,val])
        return all_combinations


    def _build_ANN(self,state_size,hl_size,action_size,learning_rate):    
        # Defining network model
        model = keras.Sequential()
        try:
            model.add(keras.layers.Dense(hl_size[0],input_shape=(state_size[0],state_size[1]),activation='relu'))
            for i in hl_size:
                model.add(keras.layers.Dense(i,activation='relu'))
        except IndexError: # Zero hidden layer
            model.add(keras.layers.Dense(len(self.action_choices),input_shape=(state_size[0],state_size[1]),activation='relu'))
        model.add(keras.layers.Dense(len(self.action_choices)))
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
        return model        

    def remember(self, state, action, next_state,reward,done):
        self.memory.append((state, action, next_state, reward,done))
        self.replay_counter += 1

    def act_greedy(self,state):
        pred = self.ANN_model.predict(state) 
        choice = np.where(pred[0][0]==max(pred[0][0]))[0][0]
        return choice
         
    def act(self, states):
        if np.random.rand() <= self.epsilon: # Exploration 
            random_action = random.randint(0,len(self.action_choices)-1)
            return random_action
        return self.act_greedy(states) # Exploitation

    def is_ready(self,batch_size):
        if len(self.memory)< batch_size:
            return False
        return True

    def Qreplay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, next_state,reward,done in minibatch:
            target = reward
            # state_data = self.process_state_data(state)
            # next_state_data = self.process_state_data(next_state)
            if not done:
                target = (reward + self.gamma *np.amax(self.ANN_model.predict(next_state)))
            target_f = self.ANN_model.predict(state)
            target_f[0][0][action] = target
            self.ANN_model.fit(state, target_f, epochs=1, verbose=0)
        self.decay_exploration()
        # self.replay_counter = 0

    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay