from collections import deque
from tensorflow import keras
from params import *
import numpy as np 
import random
class Agent():
    def __init__(self,
            hl_size=[10,10],
            state_size=OBSERVATIONS*2-1, #All states plus the gradients
            action_size=VALVE_POSITIONS
        ):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.memory = deque(maxlen=2000) 
        
        self.valve_positions = self.get_valve_positions(action_size)
        self.ANN_model = self.build_ANN(state_size,hl_size,action_size,learning_rate=0.01)
        
    
    def build_ANN(self,state_size,hl_size,action_size,learning_rate):    
        # Defining network model
        model = keras.Sequential()
        model.add(keras.layers.Dense(hl_size[0],input_shape=(state_size,),activation='relu'))
        for i in hl_size:
            model.add(keras.layers.Dense(i,activation='relu'))
        model.add(keras.layers.Dense(action_size,activation='softmax'))
        
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
        return model
    def get_valve_positions(self,action_size):
        valve_positions= []
        for i in range(action_size):
            valve_positions.append(i/(action_size-1))
        return np.array(valve_positions)

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def predict(self,states):
        states_grad = [states[i+1]- states[i] for i in range(len(states[:-1]))] # calculate dhdt
        states_data = np.array(states+states_grad) # Combine level with gradient of level

        states_data = states_data.reshape(1,len(states_data))
        pred = self.ANN_model.predict(states_data) 
        choice = np.where(pred[0]==max(pred[0]))[0][0]
        z = self.valve_positions[choice]
        # random 
        z = random.uniform(0,1)
        return z

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #act_values = self.model.predict(state)
        # return np.argmax(act_values[0])  # returns action
        return random.randrange(self.action_size)
