# Reinforcement Learning in Process Control [WIP].

<img src="https://github.com/emedd33/Reinforcement-Learning-in-Process-Control/blob/master/Q_learning/1_Tank/visualize/images/DescriptionImage.png" width="500">


#### Install requirements
Python 3.6.7 was used, not sure which versions are supported
To use python 2.7 minor tweeks to the code have to be made.

Create a virtual environment with Python 3.6+.

Run the following to install requirements for the project.
```shell
pip install -r requirements.txt
```
## The different projects
### Q_learning 
Off policy method of Deep Q networks which trains a neural network to approximate the value of beeing in different states based on series of **1 Tank**,**2 Tanks** or **6 Tanks**. All Q-learning methods uses a batch learning method.

###Policy gradint:
Work in progress

###Tank_Actor_Critic:
Coming later

#### How to run the different project and update parameters
Run main.py in each project. The different project are independent on each other.
To alter parameters change the values in python file params.py and Tank_params for the 6 tank project.



