from models.environment import Environment
from models.p_controller import P_controller
from params import MAIN_PARAMS, TANK_DIST, TANK_PARAMS, AGENT_PARAMS
from rewards import get_reward_2 as get_reward
import matplotlib.pyplot as plt
import numpy as np
import keyboard

plt.style.use("ggplot")


def main(kc=AGENT_PARAMS["KC"]):
    environment = Environment(TANK_PARAMS, TANK_DIST, MAIN_PARAMS)
    controller = P_controller(environment, AGENT_PARAMS, kc)
    init_h = TANK_PARAMS["init_level"] * TANK_PARAMS["height"]
    h = [init_h]
    z = [AGENT_PARAMS["INIT_POSITION"]]
    d = [TANK_DIST["nom_flow"]]
    reward = []
    max_time = MAIN_PARAMS["Max_time"]
    for t in range(max_time):
        new_z = controller.get_z(h[-1])
        z.append(new_z)
        new_h = environment.get_next_state(z[-1], t)
        new_reward = get_reward(h[-1] / 10, False)
        print(new_reward)
        reward.append(new_reward)

        if TANK_DIST["add"]:
            new_d = environment.model.dist.flow
            d.append(new_d)
        h.append(new_h)

        if environment.show_rendering:
            environment.render(z[-1])

        if keyboard.is_pressed("ctrl+x"):
            break
    _, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)

    ax1.plot(h[:-1], color="peru", label="Tank 1")
    ax1.set_ylim(0, 10)
    ax1.set_ylabel("Level")
    ax1.legend()

    ax2.plot(z[1:], color="peru", label="Tank 1")
    ax2.set_ylabel("Valve")
    ax2.legend()
    ax2.set_ylim(0, 1.01)

    ax3.plot(d[:-1], color="peru", label="Tank 1")
    ax3.set_ylabel("Disturbance")
    ax3.legend()

    # plt.legend([l1, l2, l3], ["Tank height", "Valve position", "Disturbance"])
    plt.tight_layout()
    plt.xlabel("Time")
    plt.show()
    return np.sum(reward)


if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max time in each episode: {}".format(MAIN_PARAMS["Max_time"]))
    reward = main()
    print(reward)
