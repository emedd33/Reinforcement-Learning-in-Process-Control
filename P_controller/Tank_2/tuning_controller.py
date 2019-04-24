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
        sys.stdout.write(
            "\r"
            + "Tank "
            + str(tank_number + 1)
            + ": Current kc iteration: "
            + str(round(kc, 2))
        )
        kc += kc_inc
        sys.stdout.flush()
    print(f"\nSimulation Done for tank {tank_number+1}")
    print("Max reward: ", max_reward, " with kc = ", round(max_reward_kc, 2))
    all_max_reward_values.append(rewards)
    all_max_rewards.append([max_reward_kc, max_reward])
    all_kc_app.append(kc_app)
    return max_reward_kc


for i in range(2):
    max_reward_kc = tune_controllers(i)
    AGENT_PARAMS_LIST[i]["KC"] = max_reward_kc

_, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)
ax1.plot(all_kc_app[0], all_max_reward_values[0], color="peru", label="Tank 1")
ax1.set_ylabel("SSE")
ax1.set_xlabel("KC")
ax1.set_ylim(top=0.01)
ax1.legend(loc="upper right")

ax2.plot(
    all_kc_app[1], all_max_reward_values[1], color="firebrick", label="Tank 2"
)
ax2.legend(loc="upper right")
ax2.set_ylabel("SSE")
ax2.set_ylim(top=0.01)
ax2.set_xlabel("KC")

plt.tight_layout()

plt.show()
print(f"Best KC for Tank 1 was {round(all_max_rewards[0][0], 2)}")
print(f"Best KC for Tank 2 was {round(all_max_rewards[1][0], 2)}")
