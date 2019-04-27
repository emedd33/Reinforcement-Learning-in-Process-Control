# Reinforcement Learning in Process Control [WIP].
![alt text](DescriptionImage.png)


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
Q Actor Critic implemented by combining REINFORCE with Q-learning.

#### How to run the different project and update parameters
Run main.py in each project. The different project are independent on each other.
To alter parameters change the values in python file params.py and Tank_params for the 6 tank project.



