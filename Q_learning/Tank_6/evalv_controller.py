from models.Agent import Agent
from models.environment import Environment
from evalv_params import MAIN_PARAMS, AGENT_PARAMS, TANK_DIST
from params import TANK_PARAMS
import os
import matplotlib.pyplot as plt
import numpy as np
from rewards import sum_rewards
from rewards import get_reward_2 as get_reward

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
    h_ = np.array([state[0][i][0] for i in range(6)])
    h.append(h_)
    for t in range(MAIN_PARAMS["MAX_TIME"]):
        action = agent.act(state[-1])  # get action choice from state
        z_ = agent.action_choices[
            action
        ]  # convert action choice into valve position
        z.append(np.array(z_))
        terminated, next_state = environment.get_next_state(
            z[-1], state[-1], t
        )  # Calculate next state with action
        reward = sum_rewards(
            next_state, terminated, get_reward
        )  # get reward from transition to next state
        # Store data
        episode_reward.append(reward)

        state.append(next_state)
        h_ = []
        d_ = []
        for i in range(agent.n_tanks):
            d_.append(
                environment.tanks[i].dist.flow[t - 1] + environment.q_inn[i]
            )
            h_.append(np.array(next_state[i][0]))
        d.append(d_)
        h.append(h_)
        if environment.show_rendering:
            environment.render(z[-1])
        if True in terminated:
            break

        if not environment.running:
            break
    colors = [
        "peru",
        "firebrick",
        "darkslategray",
        "darkviolet",
        "mediumseagreen",
        "darkcyan",
    ]
    print(f"reward: {np.sum(episode_reward)}")
    h = np.array(h) * 10
    d = np.array(d)
    z = np.array(z)
    for i in range(2):
        _, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
        ax1.plot(
            h[1:-1, 0 + i * 3],
            color=colors[0 + i * 3],
            label="Tank {}".format(str(1 + i * 3)),
        )
        ax1.plot(
            h[1:-1, 1 + i * 3],
            color=colors[1 + i * 3],
            label="Tank {}".format(str(2 + i * 3)),
        )
        ax1.plot(
            h[1:-1, 2 + i * 3],
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
        ax2.set_ylim(-0.01, 1.01)

        ax3.plot(
            d[1:-1, 0 + i * 3],
            color=colors[0 + i * 3],
            label="Tank {}".format(str(1 + i * 3)),
        )
        ax3.plot(
            d[1:-1, 1 + i * 3],
            color=colors[1 + i * 3],
            label="Tank {}".format(str(2 + i * 3)),
        )
        ax3.plot(
            d[1:-1, 2 + i * 3],
            color=colors[2 + i * 3],
            label="Tank {}".format(str(3 + i * 3)),
        )
        ax3.set_ylabel("Disturbance")
        ax3.legend(loc="upper left")
        ax3.set_ylim(-0.01, 4)

        plt.tight_layout()
        plt.xlabel("Time")
        plt.show()


if __name__ == "__main__":
    print("#### SIMULATION EVALUATION STARTED ####")
    print("  Max time in each episode: {}".format(MAIN_PARAMS["MAX_TIME"]))
    main()
