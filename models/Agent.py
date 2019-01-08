from collections import deque
from tensorflow import keras
from params import *
import numpy as np 
class Agent():
    def __init__(self,
            hidden_size=[10,10],
            input_size=OBSERVATIONS,
            output_size=VALVE_POSITIONS
        ):
        self.action_memory = deque(maxlen=3000)
        self.state_memory = deque(maxlen=3000)
        
        self.get_valve_positions(output_size)
        self.build_ANN(input_size,hidden_size,output_size,learning_rate=0.01)
    
    def build_ANN(self,input_size,hidden_size,output_size,learning_rate):    
        # Defining network model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(hidden_size[0],input_shape=(input_size-1,),activation='relu'))
        for layer in hidden_size:
            self.model.add(keras.layers.Dense(layer,activation='relu'))
        self.model.add(keras.layers.Dense(output_size,activation='softmax'))
        
        self.model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=learning_rate)
            )
    
    def get_valve_positions(self,output_size):
        valve_positions= []
        for i in range(output_size):
            valve_positions.append(i/(output_size-1))
        self.valve_positions = np.array(valve_positions)

    def predict(self,x):
        x_delta = np.array([x[i+1]- x[i] for i in range(len(x[:-1]))])
        x_delta = x_delta.reshape(1,len(x_delta)) # final inputs to model, gradient of states

        pred = self.model.predict(x_delta) 
        choice = np.where(pred[0]==max(pred[0]))[0][0]
        z = self.valve_positions[choice]
        return z
    