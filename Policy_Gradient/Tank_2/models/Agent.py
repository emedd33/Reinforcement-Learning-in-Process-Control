from collections import deque
import torch
from .Network import Net
import numpy as np


class Agent:
    def __init__(self, AGENT_PARAMS):
        "Parameters are set in the params.py file"
        self.memory = deque(maxlen=AGENT_PARAMS["MEMORY_LENGTH"])

        self.load_model = AGENT_PARAMS["LOAD_MODEL"]
        self.save_model_bool = AGENT_PARAMS["SAVE_MODEL"]
        self.train_model = AGENT_PARAMS["TRAIN_MODEL"]
        self.model_name = AGENT_PARAMS["MODEL_NAME"]

        self.n_tanks = AGENT_PARAMS["N_TANKS"]
        self.state_size = AGENT_PARAMS["OBSERVATIONS"]
        self.action_state = None
        self.actions = None
        self.z_variance = AGENT_PARAMS["Z_VARIANCE"]
        self.action_delay_cnt = [9] * self.n_tanks
        self.action_delay = AGENT_PARAMS["ACTION_DELAY"]

        if self.train_model:
            self.epsilon = [AGENT_PARAMS["EPSILON"]] * self.n_tanks
        else:
            self.epsilon = [0] * self.n_tanks

        # self.action_choices = self._build_action_choices(self.action_size)
        self.valvpos_uncertainty = AGENT_PARAMS["VALVEPOS_UNCERTAINTY"]
        self.epsilon_min = AGENT_PARAMS["EPSILON_MIN"]
        self.epsilon_decay = AGENT_PARAMS["EPSILON_DECAY"]
        self.gamma = AGENT_PARAMS["GAMMA"]
        self.buffer = 0
        self.buffer_thres = AGENT_PARAMS["BUFFER_THRESH"]

        self.learning_rate = AGENT_PARAMS["LEARNING_RATE"]
        self.hl_size = AGENT_PARAMS["HIDDEN_LAYER_SIZE"]
        self.batch_size = AGENT_PARAMS["BATCH_SIZE"]
        self.networks = self._build_network()
        self.mean_reward_memory = deque(maxlen=AGENT_PARAMS["MEMORY_LENGTH"])
        self.mean_reward_memory.append([0] * self.n_tanks)

    def _build_action_choices(self, action_size):
        "Create a list of the valve positions ranging from 0-1"
        valve_positions = []
        for i in range(action_size):
            valve_positions.append((i) / (action_size - 1))
        return np.array(list(reversed(valve_positions)))

    def _build_network(self):
        networks = []
        for i in range(self.n_tanks):
            if self.load_model:
                network = Net(
                    self.state_size,
                    self.hl_size,
                    self.learning_rate,
                )
                model_name = self.model_name
                path = (
                    "Policy_Gradient/Tank_2/saved_networks/usable_networks/"
                    + model_name
                    + ".pt"
                )
                network.load_state_dict(torch.load(path))
                network.eval()
                networks.append(network)
            else:
                network = Net(
                    self.state_size,
                    self.hl_size,
                    self.learning_rate,
                )
                networks.append(network)
        return networks

    def remember(self, states, reward, terminated, t):
        "Stores instances of each time step"
        if self.train_model:
            replay = []
            for i in range(self.n_tanks):
                if terminated[i]:
                    if len(states) <= self.action_delay[i] + 2:
                        action_state = states[i][0]
                    else:
                        action_state_index = -self.action_delay_cnt[i] - 2
                        action_state = states[action_state_index][i]
                    replay.append(
                        np.array(
                            [
                                action_state,
                                self.actions[i],
                                reward[i],
                                states[-1][i],
                                terminated[i],
                                False,
                                str(i)+"model"
                            ]
                        )
                    )
                    self.buffer += 1
                elif (
                    self.action_delay_cnt[i] >= self.action_delay[i]
                    and t >= self.action_delay[i]
                ):
                    action_state = states[-self.action_delay[i] - 2][i]
                    replay.append(
                        np.array(
                            [
                                action_state,
                                self.actions[i],
                                reward[i],
                                states[-1][i],
                                terminated[i],
                                False,
                                str(i)+"model"
                            ]
                        )
                    )
                elif True in terminated:
                    replay.append(
                        np.array(
                            [
                                [999] * self.state_size,
                                None,
                                0,
                                [999] * self.state_size,
                                True,
                                True,
                                str(i)+"model"
                            ]
                        )
                    )
            if True in terminated:
                self.memory.append(replay)
            elif not len(replay) == self.n_tanks:
                return
            else:
                self.memory.append(replay)
            self.buffer += 1

    def act(self, state):
        """
        Agent uses the state and gives either an
        action of exploration or explotation
        """
        actions = []
        for i in range(self.n_tanks):
            if self.action_delay_cnt[i] >= self.action_delay[i]:
                self.action_delay_cnt[i] = 0

                if np.random.rand() <= float(self.epsilon[i]):  # Exploration
                    random_action = np.random.uniform(0, 1)
                    actions.append(random_action)
                else:
                    action = self.act_greedy(state, i)  # Exploitation
                    actions.append(action)
            else:
                actions = self.actions
                self.action_delay_cnt[i] += 1
        self.actions = actions
        return self.actions

    def act_greedy(self, state, i):
        "Predict the optimal action to take given the current state"

        action_tensor = self.networks[i].forward(state[i])
        action = np.random.normal(action_tensor.item(), self.z_variance)
        action = 0 if action < 0 else action
        action = 1 if action > 1 else action
        return action

    def is_ready(self):
        "Check if enough data has been collected"
        if not self.train_model:  # Model has been set to not collect data
            return False
        if len(self.memory) < self.batch_size:
            return False
        if self.buffer < self.buffer_thres:
            return False
        return True

    def PolicyGradientReplay(self, e):
        """"
        Train the model to improve the predicted value of consecutive
        recurring states, Off policy Q-learning with batch training
        """
        minibatch = np.array(self.memory)
        batch_size = len(minibatch[:, 0])
        disc_rewards = []
        for j in range(self.n_tanks):
            agent_batch = minibatch[:, j]
            dummy_data = np.stack(agent_batch[:, 5])
            dummy_data_index = np.where(dummy_data)[0]

            states = np.stack(agent_batch[:, 0])
            actions = np.stack(agent_batch[:, 1])
            rewards = np.stack(agent_batch[:, 2])
            # next_states = np.stack(agent_batch[:, 3])
            # terminated = np.stack(agent_batch[:, 4])

            rewards = self.discount_rewards(rewards)
            disc_rewards.append(rewards)
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            for i in range(batch_size):
                if reward_std != 0:
                    rewards[i] = (rewards[i] - reward_mean) / reward_std
                else:
                    rewards[i] = (rewards[i] - reward_mean)

            self.networks[j].backward(
                states, actions, rewards, dummy_data_index
            )
            self.mean_reward_memory.append(disc_rewards)
            self.decay_exploration(j)
        self.memory.clear()

    def discount_rewards(self, reward):
        """ computes discounted reward """
        discounted_r = np.zeros_like(reward)
        running_add = 0
        for t in reversed(range(0, reward.size)):
            running_add = running_add * self.gamma + reward[t]

            discounted_r[t] = running_add
        return discounted_r.astype(float)

    def decay_exploration(self, j):
        "Lower the epsilon valvue to favour greedy actions"
        if self.epsilon[j] > self.epsilon_min:
            self.epsilon[j] = self.epsilon[j] * self.epsilon_decay[j]

    def reset(self, init_state):
        self.action_state = init_state[0]
        self.action = None
        self.action_delay_cnt = self.action_delay

    def save_model(self, mean_reward, max_mean_reward):
        "Save the model given a better model has been fitted"

        if mean_reward >= max_mean_reward:

            model_name = "Network_" + str(self.hl_size) + "HL"
            # + str(int(mean_reward))
            path = "Policy_Gradient/Tank_2/saved_networks/" + model_name + ".pt"
            torch.save(self.networks[0].state_dict(), path)
            print("ANN_Model was saved")
            max_mean_reward = mean_reward
        return max_mean_reward
