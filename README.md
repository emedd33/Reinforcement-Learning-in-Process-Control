# Reinforcement Learning in Process Control.

<img src="https://github.com/emedd33/Reinforcement-Learning-in-Process-Control/blob/master/Tank_Q-learning/visualize/images/DescriptionImage.png" width="500">


#### Install requirements
Python 3.6.7 was used, not sure which versions are supported
To use python 2.7 minor tweeks to the code have to be made.

Create a virtual environment with Python 3.6+.

Run the following to install requirements for the project.
```shell
pip install -r requirements.txt
```
#### The different projects
**Tank_Q-learning:** Controlling one tank by an agent which utilizes model free, value based aproximator with TD(0), default is set with gamme equals 0 so technically is not a q-learning but rather temperoal difference learning by value function approximation, feel free to change gamma.

#### How to run the different project and update parameters
Run main.py in each project. The different project are independent on each other.
To alter parameters change the values in python file params.py.

Also, Feel free to experiment with the reward function in the Environment.get_reward() function.


