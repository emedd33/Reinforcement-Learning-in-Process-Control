from main import main
import matplotlib.pyplot as plt
from params import TANK1_DIST, AGENT_PARAMS_LIST
import numpy as np
import sys

TANK1_DIST["pre_def_dist"] = False


tau_c_start = 10
tau_c_inc = 1
tau_c_end = 15
all_max_rewards = []
all_max_reward_values = []
number_of_tau_c_evaluations = 100
all_tau_c_app = []


def tune_controllers(tank_number=0):
    rewards = []
    max_reward = -99
    max_reward_tau_c = 0
    tau_c_app = []

    tau_c = tau_c_start
    while tau_c < tau_c_end:
        tau_c_app.append(tau_c)
        reward = [
            main(tau_c_tuning=tau_c, tuning_number=tank_number, plot=False)
            for i in range(number_of_tau_c_evaluations)
        ]
        rewards.append(np.mean(reward))
        if rewards[-1] > max_reward:
            max_reward = rewards[-1]
            max_reward_tau_c = tau_c_app[-1]
        sys.stdout.write(
            "\r"
            + "Tank "
            + str(tank_number + 1)
            + ": Current tau_c iteration: "
            + str(round(tau_c, 2))
        )
        tau_c += tau_c_inc
        sys.stdout.flush()
    print(f"\nSimulation Done for tank {tank_number+1}")
    print(max_reward, " with " + "tau_c: " + str(max_reward_tau_c))

    all_max_reward_values.append(rewards)
    all_max_rewards.append([max_reward_tau_c, max_reward])
    all_tau_c_app.append(tau_c_app)
    return max_reward_tau_c


for i in range(2):
    max_reward_tau_c = tune_controllers(i)
    AGENT_PARAMS_LIST[i]["TAU_C"] = max_reward_tau_c

_, (ax1, ax2) = plt.subplots(2, sharex=False, sharey=False)
ax1.plot(
    all_tau_c_app[0], all_max_reward_values[0], color="peru", label="Tank 1"
)
ax1.set_ylabel("SSE")
ax1.set_xlabel(r"$\tau_c$")
ax1.legend(loc="upper right")

ax2.plot(
    all_tau_c_app[1],
    all_max_reward_values[1],
    color="firebrick",
    label="Tank 2",
)
ax2.legend(loc="upper right")
ax2.set_ylabel("SSE")
ax2.set_xlabel(r"$\tau_c$")

plt.tight_layout()

plt.show()
print(f"Best tau_c for Tank 1 was {round(all_max_rewards[0][0], 2)}")
print(f"Best tau_c for Tank 2 was {round(all_max_rewards[1][0], 2)}")
