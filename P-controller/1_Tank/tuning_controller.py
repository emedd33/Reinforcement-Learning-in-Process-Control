from main import main
import matplotlib.pyplot as plt

kc = 0
kc_app = []
reward = []
kc_inc = 0.1
max_reward = 0
max_reward_kc = 0
while kc < 10:
    kc_app.append(kc)
    reward.append(main(kc))
    if reward[-1] > max_reward:
        max_reward = reward[-1]
        max_reward_kc = kc
    kc +=kc_inc
    print(kc)
print("Simulation Done")
print(max_reward," with kc = ", max_reward_kc)
plt.scatter(kc_app,reward)
plt.show()
