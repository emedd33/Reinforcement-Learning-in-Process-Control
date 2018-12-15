import tensorflow as tf 
import numpy as np 
import tflowtools as TFT 

class ANN():
        
    def __init__(self,structure=[1,10,10,2], learning_rate=0.1):
        self.input_size = structure[0]
        self.hidden1_size = structure[1]
        self.hidden2_size = structure[2]
        self.output_size = structure[3]
        
        self.learning_rate = learning_rate
        self.build(
            self.input_size,
            self.hidden1_size,
            self.hidden2_size,
            self.output_size
        )
        
    def build(self,ni,nh1,nh2,no):
        self.w1 = tf.Variable(np.random.uniform(-.1,.1,size=(ni,nh1)),name='Weights-1')  # first weight array
        self.w2 = tf.Variable(np.random.uniform(-.1,.1,size=(nh1,nh2)),name='Weights-2') # second weight array
        self.w3 = tf.Variable(np.random.uniform(-.1,.1,size=(nh2,no)),name='Weights-3') # second weight array

        self.b1 = tf.Variable(np.random.uniform(-.1,.1,size=nh1),name='Bias-1')  # First bias vector
        self.b2 = tf.Variable(np.random.uniform(-.1,.1,size=nh2),name='Bias-2')  # Second bias vector
        self.b3 = tf.Variable(np.random.uniform(-.1,.1,size=no),name='Bias-3')  # Second bias vector

        self.input = tf.placeholder(tf.float64,shape=(1,ni),name='Input')

        self.hidden1 = tf.nn.relu(tf.matmul(self.input,self.w1) + self.b1,name="Hidden1")
        self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1,self.w2) + self.b2,name="Hidden2")
        
        self.output = tf.nn.softmax(tf.matmul(self.hidden2,self.w3) + self.b3, name = "Outputs")
        self.target = tf.placeholder(tf.float64,shape=(1,no),name='Target')

        self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
        self.predictor = self.output  # Simple prediction runs will request the value of outputs
        
        # Defining the training operator
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    