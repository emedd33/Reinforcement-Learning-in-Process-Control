from models.Agent import Agent
from models.environment import Environment
from evalv_params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST
from params import TANK_PARAMS
import os
import matplotlib.pyplot as plt
import numpy as np
from rewards import get_reward_2 as get_reward
from rewards import sum_rewards

plt.style.use("ggplot")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    # ============= Initialize variables and objects ===========#
    environment = Environment(TANK_PARAMS, TANK_DIST, MAIN_PARAMS)
    agent = Agent(AGENT_PARAMS)
    z = []
    h = []
    d = []
    # ================= Running episodes =================#

    state, episode_reward = environment.reset()
    h_ = np.array([state[0][0][0]])
    h.append(h_)
    for t in range(MAIN_PARAMS["MAX_TIME"]):
        z_ = agent.act(state[-1])  # get action choice from state
        z.append(np.array(z_))
        terminated, next_state = environment.get_next_state(
            z[-1], state[-1], t
        )  # Calculate next state with action
        reward = sum_rewards(next_state, terminated, get_reward)  # get reward from transition to next state
        # Store data
        episode_reward.append(reward)

        state.append(next_state)
        h_ = []
        d_ = []
        for i in range(agent.n_tanks):
            d_.append(environment.tanks[i].dist.flow[t] + environment.q_inn[i])
            h_.append(np.array(next_state[i][0]))
        d.append(d_)
        h.append(h_)
        if environment.show_rendering:
            environment.render(z[-1])
        if True in terminated:
            break

        if not environment.running:
            break
    print(np.sum(episode_reward))

    _, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
    d = np.array(d)
    h = np.array(h[:-1])
    z = np.array(z)
    h *= 10

    ax1.plot(h[:-1, 0], color="peru", label="Tank 1")
    ax1.set_ylabel("Level")
    ax1.legend(loc="upper right")
    ax1.set_ylim(2.5, 7.5)

    ax2.plot(z[1:, 0], color="peru", label="Tank 1")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Valve")
    ax2.set_ylim(0, 1.01)

    ax3.plot(d[:, 0], color="peru", label="Tank 1")
    ax3.set_ylim(0, 4)
    ax3.set_ylabel("Disturbance")
    ax3.legend(loc="upper right")

    # plt.legend([l1, l2, l3], ["Tank height", "Valve position", "Disturbance"])
    plt.tight_layout()
    plt.xlabel("Time")
    plt.show()


if __name__ == "__main__":
    print("#### SIMULATION EVALUATION STARTED ####")
    print("Max time in each episode: {}".format(MAIN_PARAMS["MAX_TIME"]))
    main()
