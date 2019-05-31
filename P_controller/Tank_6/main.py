from models.environment import Environment
from models.p_controller import P_controller
from params import (
    MAIN_PARAMS,
    TANK_DIST_LIST,
    TANK_PARAMS_LIST,
    AGENT_PARAMS_LIST,
)
import matplotlib.pyplot as plt
import numpy as np
from rewards import sum_rewards
from rewards import get_reward_SSE as get_reward

plt.style.use("ggplot")


def main(tau_c_tuning=30, tuning_number=None, plot=True):
    environment = Environment(TANK_PARAMS_LIST, TANK_DIST_LIST, MAIN_PARAMS)

    controllers = []
    for i, AGENT_PARAMS in enumerate(AGENT_PARAMS_LIST):
        controller = P_controller(environment, AGENT_PARAMS, i)
        controllers.append(controller)
    if tuning_number is not None:
        controllers[tuning_number].evalv_kc(tau_c_tuning)

    init_h = []
    for tank in environment.tanks:
        init_h.append(tank.level)

    init_z = []
    for AGENT_PARAMS in AGENT_PARAMS_LIST:
        init_z.append(AGENT_PARAMS["INIT_POSITION"])

    init_d = []
    for TANK_DIST in TANK_DIST_LIST:
        init_d.append(TANK_DIST["nom_flow"])

    h = [init_h]
    z = [init_z]
    d = [init_d]

    max_time = MAIN_PARAMS["MAX_TIME"]
    episode_reward = []
    for t in range(max_time):
        new_z = []
        new_h = []
        q_out = [0]
        for i, controller in enumerate(controllers):
            new_z_ = controller.get_z(h[-1][i])
            new_z.append(new_z_)
            new_h_, q_out_ = environment.get_next_state(
                z[-1][i], i, t, q_out[i]
            )
            new_h.append(new_h_)
            q_out.append(q_out_)
        z.append(new_z)
        h.append(new_h)

        new_d = []
        for i, TANK_DIST in enumerate(TANK_DIST_LIST):
            if TANK_DIST["add"]:
                new_d_ = environment.tanks[i].dist.flow[t]
                new_d.append(new_d_ + q_out[i])
            else:
                new_d.append(q_out[i])
        d.append(new_d)

        reward = sum_rewards(
            new_h, [False], get_reward
        )  # get reward from transition to next state
        episode_reward.append(reward)
        if environment.show_rendering:
            environment.render(z[-1])
    if plot:
        colors = [
            "peru",
            "firebrick",
            "darkslategray",
            "darkviolet",
            "mediumseagreen",
            "darkcyan",
        ]
        for i in range(2):
            _, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
            h = np.array(h)
            d = np.array(d)
            z = np.array(z)
            ax1.plot(
                h[:-1, 0 + i * 3],
                color=colors[0 + i * 3],
                label="Tank {}".format(str(1 + i * 3)),
            )
            ax1.plot(
                h[:-1, 1 + i * 3],
                color=colors[1 + i * 3],
                label="Tank {}".format(str(2 + i * 3)),
            )
            ax1.plot(
                h[:-1, 2 + i * 3],
                color=colors[2 + i * 3],
                label="Tank {}".format(str(3 + i * 3)),
            )
            ax1.set_ylabel("Level")
            ax1.legend(loc="upper left")
            ax1.set_ylim(2.5, 7.5)

            ax2.plot(
                z[1:, 0 + i * 3],
                color=colors[0 + i * 3],
                label="Tank {}".format(str(1 + i * 3)),
            )
            ax2.plot(
                z[1:, 1 + i * 3],
                color=colors[1 + i * 3],
                label="Tank {}".format(str(2 + i * 3)),
            )
            ax2.plot(
                z[1:, 2 + i * 3],
                color=colors[2 + i * 3],
                label="Tank {}".format(str(3 + i * 3)),
            )
            ax2.set_ylabel("Valve")
            ax2.legend(loc="upper left")
            ax2.set_ylim(0, 1.01)

            ax3.plot(
                d[2:, 0 + i * 3],
                color=colors[0 + i * 3],
                label="Tank {}".format(str(1 + i * 3)),
            )
            ax3.plot(
                d[2:, 1 + i * 3],
                color=colors[1 + i * 3],
                label="Tank {}".format(str(2 + i * 3)),
            )
            ax3.plot(
                d[2:, 2 + i * 3],
                color=colors[2 + i * 3],
                label="Tank {}".format(str(3 + i * 3)),
            )
            ax3.set_ylabel("Disturbance")
            ax3.legend(loc="upper left")
            ax3.set_ylim(0, 4)

            plt.tight_layout()
            plt.xlabel("Time")
            plt.show()
        print(np.sum(episode_reward))
    episode_reward = np.array(episode_reward)
    return np.mean(episode_reward[:, tuning_number])


if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max time in each episode: {}".format(MAIN_PARAMS["MAX_TIME"]))
    reward = main()
