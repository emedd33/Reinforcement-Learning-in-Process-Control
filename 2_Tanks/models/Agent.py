from collections import deque
from tensorflow import keras
from params import *
import numpy as np 
import random
import itertools
class Agent():
    def __init__(self,
            hl_size=NUMBER_OF_HIDDEN_LAYERS,
            state_size=OBSERVATIONS,
            action_size=VALVE_POSITIONS,
            n_tanks = N_TANKS
        ):

        self.state_size = state_size
        self.action_size = action_size
        self.action_choices = self._set_action_choices(action_size)
        self.memory = [deque(maxlen=MEMORY_LENGTH)]*n_tanks
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON  # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.replay_counter = 0
        self.n_tanks = N_TANKS
        self.ANN_models = []
        for i in range(self.n_tanks):
            self.ANN_models.append(self._build_ANN(state_size,hl_size,action_size,learning_rate=LEARNING_RATE))
        
    def get_action_choices(self,actions):
        z = []
        for i in actions:
            z.append(self.action_choices[i])
        return z
    def _set_action_choices(self,action_size):
        valve_positions= []
        for i in range(action_size):
            valve_positions.append((i)/(action_size-1))
        return np.array(list(reversed(valve_positions)))

        # # Create a list with all the valve positions
        # all_valve_pos = []
        # for j in range(n_tanks):
        #     valve_positions= []
        #     for i in range(action_pos):
        #         valve_positions.append((i)/(action_pos-1))
        #     all_valve_pos.append(np.array(list(reversed(valve_positions))))
        
        # # Create all combinations of valve positions
        # all_combinations = []
        # for i in range(len(all_valve_pos)):
        #     arr = all_valve_pos[i]
        #     for element in arr:
        #         for j in range(i+1,len(all_valve_pos)):
        #             for val in all_valve_pos[j]:
        #                 all_combinations.append([element,val])
        # return all_combinations


    def _build_ANN(self,state_size,hl_size,action_size,learning_rate):    
        # Defining network model
        model = keras.Sequential()
        try:
            model.add(keras.layers.Dense(hl_size[0],input_shape=(self.state_size*self.n_tanks,),activation='relu'))
            for i in hl_size:
                model.add(keras.layers.Dense(i,activation='relu'))
        except IndexError: # Zero hidden layer
            model.add(keras.layers.Dense(len(self.action_choices),input_shape=(self.state_size*self.n_tanks,),activation='relu'))
        model.add(keras.layers.Dense(len(self.action_choices)))
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
        return model        

    def remember(self, state, action, next_state,reward,done):
        for i in range(self.n_tanks):
            rem_state = state.reshape(1,len(state[0][i])*self.n_tanks)
            rem_next_state = next_state.reshape(1,len(next_state[0][i])*self.n_tanks)
            self.memory[i].append((rem_state, action[i], rem_next_state, reward,done))
        self.replay_counter += 1

    def act_greedy(self,states,i):
        pred_state = states[0].reshape(1,self.n_tanks*len(states[0]))
        pred = self.ANN_models[i].predict(pred_state) 
        choice = np.where(pred[0]==max(pred[0]))[0][0]
        return choice
         
    def act(self, states):
        actions = []
        if np.random.rand() <= self.epsilon: # Exploration 
            for i in range(self.n_tanks):
                random_action = random.randint(0,len(self.action_choices)-1)
                actions.append(random_action)
            return actions
        for i in range(self.n_tanks):
            actions.append(self.act_greedy(states,i))
        return actions # Exploitation

    def is_ready(self,batch_size):
        if len(self.memory[0])< batch_size:
            return False
        return True

    def Qreplay(self, batch_size):
        for i in range(self.n_tanks):
            minibatch = random.sample(self.memory[i], batch_size)
            for state, action, next_state,reward,done in minibatch:
                target = reward

                if not done:
                    target = (reward + self.gamma *np.amax(self.ANN_models[i].predict(next_state)))
                target_f = self.ANN_models[i].predict(state)
                target_f[0][action] = target
                self.ANN_models[i].fit(state, target_f, epochs=1, verbose=0)
            self.decay_exploration()
            # self.replay_counter = 0

    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay