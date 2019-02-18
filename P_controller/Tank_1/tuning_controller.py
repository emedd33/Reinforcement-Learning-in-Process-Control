import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('C:/Users/eskil/Google Drive/Skolearbeid/5. klasse/Master')
from P_controller.Tank_1.main import main
import matplotlib.pyplot as plt
from P_controller.Tank_1.params import MAIN_PARAMS

kc = 0
kc_app = []
reward = []
kc_inc = 0.005
min_reward = 99
min_reward_kc = 0
while kc < 0.5:
    kc_app.append(kc)
    kc_reward = main(kc)
    reward.append(kc_reward/MAIN_PARAMS['Max_time'])
    if reward[-1] < min_reward:
        min_reward = reward[-1]
        min_reward_kc = kc_app[-1]
    kc +=kc_inc
    print(kc)
print("Simulation Done")
print(min_reward," with kc = ", min_reward_kc)
plt.scatter(kc_app,reward)
plt.ylabel("MSE")
plt.xlabel("KC")
plt.show()
