from main import main
import matplotlib.pyplot as plt
from params import TANK1_DIST, AGENT_PARAMS_LIST
import numpy as np
import sys

TANK1_DIST["pre_def_dist"] = False


tau_c_start = 10
tau_c_inc = 10
tau_c_end = 1000
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
            + str(round(tau_c+tau_c_inc, 2))
        )
        tau_c += tau_c_inc
        sys.stdout.flush()
    print(f"\nSimulation Done for tank {tank_number+1}")
    print("max_reward: " + str(round(max_reward, 5)) + " with " + "tau_c: " + str(max_reward_tau_c))

    all_max_reward_values.append(rewards)
    all_max_rewards.append([max_reward_tau_c, max_reward])
    all_tau_c_app.append(tau_c_app)
    return max_reward_tau_c


for i in range(6):
    max_reward_tau_c = tune_controllers(i)
    AGENT_PARAMS_LIST[i]["TAU_C"] = max_reward_tau_c

plt.subplot(3, 1, 1)
plt.plot(
    all_tau_c_app[0], all_max_reward_values[0], color="peru", label="Tank 1"
)
plt.ylabel("SSE")
plt.xlabel(r"$\tau_c$")
# plt.ylim(top=0.05)
plt.legend(loc="upper right")

plt.subplot(3, 1, 2)
plt.plot(
    all_tau_c_app[1],
    all_max_reward_values[1],
    color="firebrick",
    label="Tank 2",
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
# plt.ylim(top=0.05)
plt.xlabel(r"$\tau_c$")

plt.subplot(3, 1, 3)
plt.plot(
    all_tau_c_app[2],
    all_max_reward_values[2],
    color="firebrick",
    label="Tank 3",
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
# plt.ylim(top=0.05)
plt.xlabel(r"$\tau_c$")

plt.tight_layout()
plt.show()


plt.subplot(3, 1, 1)
plt.plot(
    all_tau_c_app[3],
    all_max_reward_values[3],
    color="firebrick",
    label="Tank 4",
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
# plt.ylim(top=0.05)
plt.xlabel(r"$\tau_c$")

plt.subplot(3, 1, 2)
plt.plot(
    all_tau_c_app[4],
    all_max_reward_values[4],
    color="firebrick",
    label="Tank 5",
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
# plt.ylim(top=0.1)
plt.xlabel(r"$\tau_c$")

plt.subplot(3, 1, 3)
plt.plot(
    all_tau_c_app[5],
    all_max_reward_values[5],
    color="firebrick",
    label="Tank 6",
)
plt.legend(loc="upper right")
plt.ylabel("SSE")
# plt.ylim(top=0.1)
plt.xlabel(r"$\tau_c$")

plt.tight_layout()
plt.show()
for i in range(6):
    print(f"Best tau_c for Tank {i+1} was {round(all_max_rewards[i][0], 2)}")
