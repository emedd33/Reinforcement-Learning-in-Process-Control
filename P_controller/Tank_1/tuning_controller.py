from main import main
import matplotlib.pyplot as plt
from params import TANK1_DIST, AGENT_PARAMS_LIST
import numpy as np
import sys

TANK1_DIST["pre_def_dist"] = False

kc_start = 0
kc_inc = 0.01
kc_end = 5
all_max_rewards = []
all_max_reward_values = []
number_of_kc_evaluations = 30
all_kc_app = []


def tune_controllers(tank_number=0):
    rewards = []
    max_reward = -99
    max_reward_kc = 0
    kc_app = []

    kc = kc_start
    while kc < kc_end:
        kc_app.append(kc)
        reward = [
            main(kc_tuning=kc, tuning_number=tank_number, plot=False)
            for i in range(number_of_kc_evaluations)
        ]
        rewards.append(np.mean(reward))
        if rewards[-1] > max_reward:
            max_reward = rewards[-1]
            max_reward_kc = kc_app[-1]
        kc += kc_inc
        sys.stdout.write(
            "\r"
            + "Tank "
            + str(tank_number + 1)
            + ": Current kc iteration: "
            + str(round(kc, 2))
        )
        sys.stdout.flush()
    print(f"\nSimulation Done for tank {tank_number+1}")
    print(max_reward, " with kc = ", max_reward_kc)
    all_max_reward_values.append(rewards)
    all_max_rewards.append([max_reward_kc, max_reward])
    all_kc_app.append(kc_app)
    return max_reward_kc


for i in range(1):
    max_reward_kc = tune_controllers(i)
    AGENT_PARAMS_LIST[i]["KC"] = max_reward_kc

_, (ax1) = plt.subplots(1, sharex=False, sharey=False)
ax1.plot(all_kc_app[0], all_max_reward_values[0], color="peru", label="Tank 1")
ax1.set_ylabel("SSE")
ax1.set_ylim(-1, 0.1)
ax1.legend(loc="upper right")
ax1.set_xlabel("KC")

plt.tight_layout()
plt.show()
print(f"Best KC for Tank 1 was {round(all_max_rewards[0][0], 2)}")
