# Reinforcement Learning in Process Control .
![alt text](DescriptionImage.png)

### Motivation 
The project was to see if the RL method from machine learning could be of use as control model for industrial systems. Replacing traditional controllers like P-controller and MPC. So this is more of a POC to see if its viable to throw a RL algorithm on a industrial system which needs to be controlled. 

The motivation is that some large complex industrial systems have have model sequations which need to be solved in order to have a control-model. And the solution is based on the system equations. Sometime the solution is hard to converge and solve. So this project was to figure out if one could give the system to the Machine, without any knowledge about the system and see if it could learn to control the systemstate given an disturbance to the system. 

### Abstraction from the thesis
Using reinforcement learning as controllers in the process industries was ex-
plored as an alternate path of doing control compared to the regular controllers.
Methods such as value-based and policy-based methods were used as controllers
for three different cases of tank level regulation. The controllers were compared
to a traditional P-controller for evaluation of the controller performance. The
reinforcement learning controllers showed promising results as they managed
to control the liquid level between the predetermined constraints. However,
the P-controllers proved a better performance with smaller input changes com-
pared to the reinforcement learning controllers which had large input changes
that resulted in oscillatory liquid level. This thesis shows that the creation
of reinforcement learning controllers is complicated and time-consuming and a
well-tuned controller would most likely perform better. However, with more re-
search and standardized approaches, there is a huge potential of including this
field into the process industries due to its ability to handle nonlinearity and long
term evaluations.

#### Install requirements
Python 3.6.7 was used, not sure which versions are supported
To use python 2.7 minor tweeks to the code have to be made.

Create a virtual environment with Python 3.6+.

Run the following to install requirements for the project.
```shell
pip install -r requirements.txt
```
requirements.txt does not include pytorch version. Go to (https://pytorch.org/get-started/locally/) and install your correct pytorch version
## The different projects
### Off policy value method (DQN):
Off policy method of Deep Q networks which trains a neural network to approximate the value of beeing in different states based on series of **1 Tank**, **2 Tanks** or **6 Tanks**. All Q-learning methods uses a batch learning method. For 2_Tank and 6_Tank, multiple agents are implemented where each agent have included the the action of the previous tanks agent as state input.

### Policy gradint method (REINFORCE):
REINFORCE Monte Carlo option of using baseline. **1 Tank**, **2 Tanks** or **6 Tanks**. All methods uses a batch learning method. For 2_Tank and 6_Tank, multiple agents are implemented where each agent have included the the action of the previous tanks agent as state input.


### Actor critic method (A2C)
Q Actor Critic implemented by combining REINFORCE with Q-learning. The method is not fully optimized so it may have some errors. Only implemented for 1 Tank

#### How to run the different project and update parameters
Run main.py in each project. The different project are independent on each other.
To alter parameters change the values in python file params.py and Tank_params for the 6 tank project.
The evalv_controller.py script is a one episode run using the predetermined disturbance_200.csv and plots the history of the valve position, liquid level and disturbance to be used for comparison of different controllers



