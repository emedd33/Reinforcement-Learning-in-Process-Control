from collections import deque
from tensorflow import keras
from params import *
import numpy as np 
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

        self.action_memory = deque(maxlen=3000) 
        self.state_memory = deque(maxlen=3000) 
        
        self.get_valve_positions(output_size)
        self.build_ANN(state_size,hl_size,action_size,learning_rate=0.01)
    
    def build_ANN(self,state_size,hl_size,action_size,learning_rate):    
        # Defining network model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(hl_size[0],input_shape=(state_size,),activation='relu'))
        for i in hl_size:
            self.model.add(keras.layers.Dense(i,activation='relu'))
        self.model.add(keras.layers.Dense(action_size,activation='softmax'))
        
        self.model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
    
    def get_valve_positions(self,action_size):
        valve_positions= []
        for i in range(action_size):
            valve_positions.append(i/(action_size-1))
        self.valve_positions = np.array(valve_positions)

    def predict(self,x):
        x_grad = [x[i+1]- x[i] for i in range(len(x[:-1]))] # calculate dhdt
        x_data = np.array(x+x_grad) # Combine level with gradient of level

        x_data = x_data.reshape(1,len(x_data))
        pred = self.model.predict(x_data) 
        choice = np.where(pred[0]==max(pred[0]))[0][0]
        z = self.valve_positions[choice]
        return z
    