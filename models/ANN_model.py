import tensorflow as tf 
from tensorflow import keras
import numpy as np 

class ANN_model():
    def __init__(self,input_size,hidden_size,output_size,max_state,learning_rate=0.01):
        self.max_state_value = max_state
        self.get_valve_positions(output_size)
        # Defining network model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(hidden_size[0],input_shape=(input_size,),activation='relu'))
        for layer in hidden_size:
            self.model.add(keras.layers.Dense(layer,activation='relu'))
        self.model.add(keras.layers.Dense(output_size,activation='softmax'))
        self.model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(lr=learning_rate))
    
    def predict(self,x):
        x = np.array(x)/self.max_state_value # Preprocesses inputs 
        x_delta = np.array([x[i+1]- x[i] for i in range(len(x[:-1]))])
        x_delta = x_delta.reshape(1,len(x_delta)) # final inputs to model, gradient of states

        pred = self.model.predict(x_delta) 
        choice = np.where(pred[0]==max(pred[0]))[0][0]
        z = self.valve_positions[choice]
        return z

    def get_valve_positions(self,output_size):
        valve_positions= []
        for i in range(output_size):
            valve_positions.append(i/(output_size-1))
        self.valve_positions = np.array(valve_positions)

        
        
    def get_z_from_prediction(self,pred):
        pred = np.array(pred)
        index = np.where(pred[0]==max(pred[0]))[0][0]
        
        
