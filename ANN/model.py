import tensorflow as tf 
import np as np 
import tflowtools as TFT 

class NeuralNetowork():
    def __init__(nh):
        self.input_size = 2**nh # size of input and output layer
        self.w1 = tf.Variable(np.random.unifrom(-0.1,0.1,size))
        
