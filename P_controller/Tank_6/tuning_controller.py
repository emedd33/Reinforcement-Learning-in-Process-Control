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
number_of_kc_evaluations = 50
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
        sys.stdout.write(
            "\r"
            + "Tank "
            + str(tank_number + 1)
            + ": Current kc iteration: "
            + str(round(kc, 2))
        )
        sys.stdout.flush()
        kc += kc_inc
    print(f"\nSimulation Done for tank {tank_number+1}")
    print("Max reward: ", max_reward, " with kc = ", round(max_reward_kc, 2))
    all_max_reward_values.append(rewards)
    all_max_rewards.append([max_reward_kc, max_reward])
    all_kc_app.append(kc_app)
    return max_reward_kc


for i in range(6):
    max_reward_kc = tune_controllers(i)
    AGENT_PARAMS_LIST[i]["KC"] = max_reward_kc

plt.subplot(3, 2, 1)
plt.plot(all_kc_app[0], all_max_reward_values[0], color="peru", label="Tank 1")
plt.ylabel("SSE")
plt.xlabel("KC")
plt.ylim(top=0.05)
plt.legend(loc="upper right")

plt.subplot(3, 2, 2)
plt.plot(
    all_kc_app[1], all_max_reward_values[1], color="firebrick", label="Tank 2"
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
plt.ylim(top=0.05)
plt.xlabel("KC")

plt.subplot(3, 2, 3)
plt.plot(
    all_kc_app[2], all_max_reward_values[2], color="firebrick", label="Tank 3"
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
plt.ylim(top=0.05)
plt.xlabel("KC")

plt.subplot(3, 2, 4)
plt.plot(
    all_kc_app[3], all_max_reward_values[3], color="firebrick", label="Tank 4"
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
plt.ylim(top=0.05)
plt.xlabel("KC")

plt.subplot(3, 2, 5)
plt.plot(
    all_kc_app[4], all_max_reward_values[4], color="firebrick", label="Tank 5"
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
plt.ylim(top=0.1)
plt.xlabel("KC")

plt.subplot(3, 2, 6)
plt.plot(
    all_kc_app[5], all_max_reward_values[5], color="firebrick", label="Tank 6"
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
plt.ylim(top=0.1)
plt.xlabel("KC")

plt.tight_layout()
plt.show()
for i in range(6):
    print(f"Best KC for Tank {i+1} was {round(all_max_rewards[i][0], 2)}")
