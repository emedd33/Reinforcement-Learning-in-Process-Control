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
            state_size=OBSERVATIONS,
            action_size=VALVE_POSITIONS,
        ): 
        "Parameters are set in the params.py file"

        self.state_size = state_size
        self.action_size = action_size
        self.action_choices = self._get_action_choices(action_size)
        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.gamma = GAMMA    # discount rate
        self.epsilon = EPSILON if TRAIN_MODEL else EPSILON_MIN # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.buffer = 0
        self.buffer_thres = BATCH_SIZE*4
        self.ANN_model = self._build_ANN(state_size,hl_size,action_size,learning_rate=0.01)
        
    
    def _get_action_choices(self,action_size):
        "Create a list of the valve positions ranging from 0-1 with acrion_size positiong"
        valve_positions= []
        for i in range(action_size):
            valve_positions.append((i)/(action_size-1))
        return np.array(list(reversed(valve_positions)))

    def _build_ANN(self,state_size,hl_size,action_size,learning_rate):  
        "Creates or loads a ANN valve function approximator"  
        
        #Load existing model from saved_models folder
        if LOAD_ANN_MODEL:
            model_name = "/ANN_"+ str(NUMBER_OF_HIDDEN_LAYERS)+"HL" 
            model_path = "Q_learning/1_Tank/saved_models" + model_name+ ".h5"
            model = keras.models.load_model(model_path)
            return model
        
        #Create new un-trained model 
        model = keras.Sequential()
        try:
            model.add(keras.layers.Dense(hl_size[0],input_shape=(state_size,),activation='relu'))
            for i in hl_size:
                model.add(keras.layers.Dense(i,activation='relu'))
        except IndexError: # Zero hidden layer
            model.add(keras.layers.Dense(action_size,input_shape=(state_size,),activation='relu'))
        model.add(keras.layers.Dense(action_size))
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
        return model        

    def remember(self, state, action, next_state,reward,done):
        "Stores instances of each time step"

        if TRAIN_MODEL:
            next_state=np.array(next_state)
            next_state=next_state.reshape(1,next_state.size)
            self.memory.append((state, action, next_state, reward,done))
            self.buffer += 1

    def act_greedy(self,state):
        "Predict the optimal action to take given the current state"

        pred = self.ANN_model.predict(state) 
        choice = np.where(pred[0]==max(pred[0]))[0][0]
        return choice
         
    def act(self, states):
        "Agent uses the state and gives either an action of exploration or explotation"
        
        if np.random.rand() <= self.epsilon: # Exploration 
            random_action = random.randint(0,self.action_size-1)
            return random_action
        return self.act_greedy(states) # Exploitation

    def is_ready(self,batch_size):
        "Check if enough data has been collected"
        
        if not TRAIN_MODEL: #Model has been set to not collect data
            return False
        if len(self.memory)< batch_size:
            return False
        if self.buffer < self.buffer_thres:
            return False
        return True

    def Qreplay(self, batch_size):
        "Train the model to improve the predicted value of consecutive recurring states, Off policy Q-learning with batch training"
        
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, next_state,reward,done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.ANN_model.predict(next_state)[0]))
            target_f = self.ANN_model.predict(state)
            target_f[0][action] = target 

            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.ANN_model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        
        self.decay_exploration()

    def decay_exploration(self):
        "Lower the epsilon valvue to favour greedy actions"

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self,mean_reward,max_mean_reward):
        "Save the model given a better model has been fitted"

        if mean_reward >= max_mean_reward:
                model_name = "/ANN_"+ str(NUMBER_OF_HIDDEN_LAYERS)+"HL"
                model_path = "Q_learning/1_Tank/saved_models" + model_name+ ".h5"
                self.ANN_model.save(model_path)
                print("ANN_Model was saved")
                max_mean_reward = mean_reward
        return max_mean_reward