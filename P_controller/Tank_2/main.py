from models.environment import Environment
from models.p_controller import P_controller
from params import (
    MAIN_PARAMS,
    TANK_DIST_LIST,
    TANK_PARAMS_LIST,
    AGENT_PARAMS_LIST,
)
import matplotlib.pyplot as plt
import keyboard
import numpy as np

plt.style.use("ggplot")


def main(kc=0.16):
    environment = Environment(TANK_PARAMS_LIST, TANK_DIST_LIST, MAIN_PARAMS)
    controllers = []
    for i, AGENT_PARAMS in enumerate(AGENT_PARAMS_LIST):
        if i == 1:
            controller = P_controller(environment, AGENT_PARAMS, i, kc)
        else:
            controller = P_controller(environment, AGENT_PARAMS, i, kc=0.16)
        controllers.append(controller)
    init_h = []
    for TANK_PARAMS in TANK_PARAMS_LIST:
        level = TANK_PARAMS["init_level"] * TANK_PARAMS["height"]
        init_h.append(level)
    init_z = []
    for AGENT_PARAMS in AGENT_PARAMS_LIST:
        init_z.append(AGENT_PARAMS["INIT_POSITION"])
    init_d = []
    for TANK_DIST in TANK_DIST_LIST:
        init_d.append(TANK_DIST["nom_flow"])

    h = [init_h]
    z = [init_z]
    d = [init_d]
    max_time = MAIN_PARAMS["Max_time"]
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
        # new_reward = environment.get_reward(h[-1])
        # reward.append(new_reward)
        new_d = []
        for i, TANK_DIST in enumerate(TANK_DIST_LIST):
            if TANK_DIST["add"]:
                new_d_ = environment.model[i].dist.flow
                new_d.append(new_d_ + q_out[i])
            else:
                new_d.append(q_out[i])
        d.append(new_d)

        if environment.show_rendering:
            environment.render(z[-1])

        if keyboard.is_pressed("ctrl+x"):
            break
    _, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
    h = np.array(h)
    d = np.array(d)
    z = np.array(z)
    ax1.plot(h[:-1, 0], color="peru", label="Tank 1")
    ax1.plot(h[:-1, 1], color="firebrick", label="Tank 2")
    ax1.set_ylabel("Level")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 10)

    ax2.plot(z[1:, 1], color="peru", label="Tank 1")
    ax2.plot(z[1:, 0], color="firebrick", label="Tank 2")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Valve")
    ax2.set_ylim(-0.01, 1.01)

    ax3.plot(d[:-1, 0], color="peru", label="Tank 1")
    ax3.plot(d[:-1, 1], color="firebrick", label="Tank 2")
    ax3.set_ylabel("Disturbance")
    ax3.legend(loc="upper right")

    # plt.legend([l1, l2, l3], ["Tank height", "Valve position", "Disturbance"])
    plt.tight_layout()
    plt.xlabel("Time")
    plt.show()
    return h


if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max time in each episode: {}".format(MAIN_PARAMS["Max_time"]))
    reward = main()
