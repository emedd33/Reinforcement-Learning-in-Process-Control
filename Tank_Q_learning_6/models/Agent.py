from collections import deque
from tensorflow import keras
from params import N_TANKS,NUMBER_OF_HIDDEN_LAYERS,OBSERVATIONS,VALVE_POSITIONS,\
    MEMORY_LENGTH,GAMMA,EPSILON,EPSILON_MIN,EPSILON_DECAY,LEARNING_RATE,LOAD_ANN_MODEL,\
        LOAD_ANN_MODEL,BATCH_SIZE,TRAIN_MODEL

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
        self.epsilon = EPSILON if TRAIN_MODEL else EPSILON_MIN # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.buffer = 0
        self.buffer_thresh = BATCH_SIZE*4
        self.n_tanks = N_TANKS
        self.ANN_models = []
        for i in range(self.n_tanks):
            self.ANN_models.append(self._build_ANN(state_size,hl_size,action_size,learning_rate=LEARNING_RATE,id=i))
        
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


    def _build_ANN(self,state_size,hl_size,action_size,learning_rate,id):    
        # Defining network model
        if LOAD_ANN_MODEL:
            model_name = "/ANN_"+ str(NUMBER_OF_HIDDEN_LAYERS)+"HL_" + str(id) 
            model_path = "Tank_2_Q_learning/models/saved_models" + model_name+ ".h5"
            model = keras.models.load_model(model_path)
            return model
        model = keras.Sequential()
        try:
            model.add(keras.layers.Dense(hl_size[0],input_shape=(OBSERVATIONS,),activation='relu'))
            for i in hl_size:
                model.add(keras.layers.Dense(i,activation='relu'))
        except IndexError: # Zero hidden layer
            model.add(keras.layers.Dense(len(self.action_choices),input_shape=(OBSERVATIONS,),activation='relu'))
        model.add(keras.layers.Dense(len(self.action_choices)))
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
        return model        

    def remember(self, state, action, next_state,reward,terminated):
        if TRAIN_MODEL:
            for i in range(self.n_tanks):
                rem_state = state[0][i]
                rem_state = rem_state.reshape(1,len(rem_state))
                rem_next_state = next_state[0][i]
                rem_next_state = rem_next_state.reshape(1,len(rem_next_state))
                self.memory[i].append((rem_state, action[i], rem_next_state, reward[i],terminated[i]))
            self.buffer += 1

    def act_greedy(self,states,i):
        pred_state = states[0][i]
        pred_state = pred_state.reshape(1,len(pred_state))
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
        if not TRAIN_MODEL:
            return False
        if len(self.memory[0])< batch_size:
            return False
        if self.buffer < self.buffer_thresh:
            return False
        return True

    def Qreplay(self, batch_size):
        loss = []
        for i in range(self.n_tanks):
            minibatch = random.sample(self.memory[i], batch_size)
            states, targets_f = [], []
            for state, action, next_state,reward,done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                            np.amax(self.ANN_models[i].predict(next_state)[0]))
                target_f = self.ANN_models[i].predict(state)
                target_f[0][action] = target 
                # Filtering out states and targets for training
                states.append(state[0])
                targets_f.append(target_f[0])
            history = self.ANN_models[i].fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        
        loss.append(history.history['loss'][0])
        self.decay_exploration()
        self.buffer = 0
        return loss

    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay