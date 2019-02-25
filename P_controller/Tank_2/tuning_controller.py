from main import main
import matplotlib.pyplot as plt
from params import MAIN_PARAMS


def get_reward(h):
    reward = 0
    for time in h:
        reward += -(time[-1] - 0.5) ** 2

    return reward


kc = 0
kc_app = []
rewards = []
kc_inc = 0.01
min_reward = -99
min_reward_kc = 0
while kc < 2:
    kc_app.append(kc)
    h = main(kc)
    reward = get_reward(h)
    rewards.append(reward / MAIN_PARAMS["Max_time"])
    if reward < min_reward:
        min_reward = reward
        min_reward_kc = kc_app[-1]
    kc += kc_inc
    print(kc)
print("Simulation Done")
print(min_reward, " with kc = ", min_reward_kc)
plt.scatter(kc_app, rewards)
plt.ylabel("MSE")
plt.xlabel("KC")
plt.show()
