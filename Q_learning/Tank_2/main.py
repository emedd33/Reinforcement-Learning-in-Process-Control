# Add the ptdraft folder path to the sys.path list
# sys.path.append("C:/Users/eskil/Google Drive/Skolearbeid/5. klasse/Master")
from models.environment import Environment
from models.Agent import Agent
from params import MAIN_PARAMS, AGENT_PARAMS, TANK_PARAMS, TANK_DIST
import os
import matplotlib.pyplot as plt
import numpy as np
from rewards import get_reward_2 as get_reward
from rewards import sum_rewards

plt.style.use("ggplot")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    # ============= Initialize variables and objects ===========#
    max_mean_reward = MAIN_PARAMS["MAX_MEAN_REWARD"]
    environment = Environment(TANK_PARAMS, TANK_DIST, MAIN_PARAMS)
    agent = Agent(AGENT_PARAMS)
    mean_episode = MAIN_PARAMS["MEAN_EPISODE"]
    episodes = MAIN_PARAMS["EPISODES"]

    all_rewards = []
    all_mean_rewards = []
    t_mean = []

    # ================= Running episodes =================#
    try:
        for e in range(episodes):
            states, episode_reward = environment.reset()  # Reset level in tank
            for t in range(MAIN_PARAMS["MAX_TIME"]):
                actions = agent.act(states[-1])  # get action choice from state
                z = agent.get_z(actions)

                terminated, next_state = environment.get_next_state(
                    z, states[-1], t
                )  # Calculate next state with action
                rewards = sum_rewards(
                    next_state, terminated, get_reward
                )  # get reward from transition to next state

                # Store data
                rewards = sum_rewards(next_state, terminated, get_reward)
                rewards.append(np.sum(rewards))
                episode_reward.append(rewards)

                states.append(next_state)
                agent.remember(states, rewards, terminated, t)

                if environment.show_rendering:
                    environment.render(z)
                if True in terminated:
                    break

            episode_reward = np.array(episode_reward)
            episode_total_reward = []
            t_mean.append(t)
            for i in range(environment.n_tanks + 1):
                episode_total_reward.append(sum(episode_reward[:, i]))
            all_rewards.append(episode_total_reward)

            # Print mean reward and save better models
            if e % mean_episode == 0 and e != 0:
                mean_reward = np.array(all_rewards[-mean_episode:])
                mean_r = []
                t_mean = int(np.mean(t_mean))
                for i in range(environment.n_tanks + 1):
                    mean_r.append(np.mean(mean_reward[:, i]))
                all_mean_rewards.append(mean_r)
                print(
                    f"Mean {mean_episode} of {e}/{episodes} episodes ### timestep {t_mean+1} ### tot reward: {mean_r[-1]}  \
                    r1: {mean_r[0]} ex1: {round(agent.epsilon[0],2)} r2: {mean_r[1]} ex2: {round(agent.epsilon[1],2)}"
                )
                t_mean = []
                if mean_r[-1] >= max_mean_reward:
                    agent.save_trained_model()
                    max_mean_reward = mean_r[-1]
                # Train model
            if agent.is_ready():
                agent.Qreplay(e)

            if not environment.running:
                break
            # if agent.epsilon <= agent.epsilon_min:
            #     break
    except KeyboardInterrupt:
        pass
    print("Memory length: {}".format(len(agent.memory)))
    print("##### {} EPISODES DONE #####".format(e + 1))
    print("Max rewards for all episodes: {}".format(np.max(all_rewards)))

    all_mean_rewards = np.array(all_mean_rewards)
    labels = ["Tank 1", "Tank 2"]

    for i in range(environment.n_tanks):
        plt.plot(all_mean_rewards[:, i], label=labels[i])
    plt.legend()
    plt.show()

    plt.plot(all_mean_rewards[:, -1], label="Total rewards")
    plt.ylabel("Mean rewards of last {} episodes".format(mean_episode))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("#### SIMULATION STARTED ####")
    print("  Max number of episodes: {}".format(MAIN_PARAMS["EPISODES"]))
    print("  Max time in each episode: {}".format(MAIN_PARAMS["MAX_TIME"]))
    print(
        "  {}Rendring simulation ".format(
            "" if MAIN_PARAMS["RENDER"] else "Not "
        )
    )
    main()
