from collections import deque
import torch
from .Network import Net
import numpy as np


class Agent:
    def __init__(self, AGENT_PARAMS):
        "Parameters are set in the params.py file"
        self.memory = deque(maxlen=AGENT_PARAMS["MEMORY_LENGTH"])

        self.load_model = AGENT_PARAMS["LOAD_MODEL"]
        self.actor_model_name = AGENT_PARAMS["LOAD_ACTOR_NAME"]
        self.critic_model_name = AGENT_PARAMS["LOAD_CRITIC_NAME"]
        self.save_model = AGENT_PARAMS["SAVE_MODEL"]
        self.train_model = AGENT_PARAMS["TRAIN_MODEL"]

        self.load_model_path = AGENT_PARAMS["LOAD_MODEL_PATH"]
        self.save_model_path = AGENT_PARAMS["SAVE_MODEL_PATH"]

        self.n_tanks = AGENT_PARAMS["N_TANKS"]
        self.state_size = AGENT_PARAMS["OBSERVATIONS"]
        self.action_state = None
        self.actions = None
        self.z_variance = AGENT_PARAMS["Z_VARIANCE"]
        self.action_delay_cnt = [9] * self.n_tanks
        self.action_delay = AGENT_PARAMS["ACTION_DELAY"]

        self.epsilon = AGENT_PARAMS["EPSILON"]

        self.valvpos_uncertainty = AGENT_PARAMS["VALVEPOS_UNCERTAINTY"]
        self.epsilon_min = AGENT_PARAMS["EPSILON_MIN"]
        self.epsilon_decay = AGENT_PARAMS["EPSILON_DECAY"]
        self.gamma = AGENT_PARAMS["GAMMA"]

        self.base_line = [
            deque(maxlen=AGENT_PARAMS["BATCH_SIZE"])
        ] * self.n_tanks

        self.actor_learning_rate = AGENT_PARAMS["ACTOR_LEARNING_RATE"]
        self.critic_learning_rate = AGENT_PARAMS["CRITIC_LEARNING_RATE"]
        self.hl_size = AGENT_PARAMS["HIDDEN_LAYER_SIZE"]
        self.batch_size = AGENT_PARAMS["BATCH_SIZE"]
        self.actors, self.critics = self._build_network()
        self.mean_reward_memory = deque(maxlen=AGENT_PARAMS["MEMORY_LENGTH"])
        self.mean_reward_memory.append([0] * self.n_tanks)

    def _build_network(self):
        actors, critics = [], []
        for i in range(self.n_tanks):
            if self.load_model[i]:
                actor = Net(
                    self.state_size, self.hl_size[i], self.actor_learning_rate[i]
                )
                critic = Net(
                    self.state_size, self.hl_size[i], self.critic_learning_rate[i]
                )
                actor_model_name = self.actor_model_name[i] + str(i)
                actor_path = self.load_model_path + actor_model_name + ".pt"
                actor.load_state_dict(torch.load(actor_path))
                actor.eval()

                critic_model_name = self.critic_model_name[i] + str(i)
                critic_path = self.load_model_path + critic_model_name + ".pt"
                critic.load_state_dict(torch.load(critic_path))
                critic.eval()
                actors.append(actor)
                critics.append(critic)
            else:
                actor = Net(
                    self.state_size, self.hl_size[i], self.actor_learning_rate[i]
                )
                critic = Net(
                    self.state_size, self.hl_size[i], self.critic_learning_rate[i]
                )
                actors.append(actor)
                critics.append(critic)
        return actors, critics

    def remember(self, states, reward, terminated, t):
        "Stores instances of each time step"

        replay = []
        for i in range(self.n_tanks):

            if terminated[i]:
                if len(states) <= self.action_delay[i] + 2:
                    action_state = states[0][i]
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
                            str(i) + "model",
                        ]
                    )
                )

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
                            str(i) + "model",
                        ]
                    )
                )
            elif True in terminated:

                action_state_index = -self.action_delay_cnt[i] - 2
                try:
                    action_state = states[action_state_index][i]
                except IndexError:
                    action_state = states[0][i]
                replay.append(
                    np.array(
                        [
                            action_state,
                            self.actions[i],
                            reward[i],
                            states[-1][i],
                            terminated[i],
                            False,
                            str(i) + "model",
                        ]
                    )
                )
        if True in terminated:
            self.memory.append(replay)
        elif not len(replay) == self.n_tanks:
            return
        else:
            self.memory.append(replay)

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

        action_tensor = self.actors[i].forward(state[i])
        action = np.random.normal(action_tensor.item(), self.z_variance[i])
        action = 0 if action < 0 else action
        action = 1 if action > 1 else action
        return action

    def is_ready(self):
        if len(self.memory) > self.batch_size:
            return True
        else:
            return False

    def AC_replay(self):
        """"
        Train the model to improve the predicted value of consecutive
        recurring states, Off policy Q-learning with batch training
        """
        minibatch = np.array(self.memory)
        batch_size = len(minibatch)
        # disc_rewards = []
        for j in range(self.n_tanks):
            if self.train_model[j]:
                agent_batch = minibatch[:, j]

                states = np.stack(agent_batch[:, 0])
                # actions = np.stack(agent_batch[:, 1])
                rewards = np.stack(agent_batch[:, 2])
                # next_states = np.stack(agent_batch[:, 3])
                # terminated = np.stack(agent_batch[:, 4])
                dummy_data = np.stack(agent_batch[:, 5])
                dummy_data_index = np.where(dummy_data)[0]

                values = self.critics[j].forward(states)
                log_probs = self.actors[j].forward(states)
                values_ = values.detach().numpy()
                Qvals = np.zeros_like(values_)
                Qvals[-1] = rewards[-1]
                for t in reversed(range(batch_size-1)):
                    Qval = rewards[t] + self.gamma * Qvals[t+1]
                    Qvals[t] = Qval

                self.actors[j].actor_backward(values, log_probs, Qvals, dummy_data_index)
                self.critics[j].critic_backward(values, log_probs, Qvals, dummy_data_index)

                # # disc_Q_value = self.discount_rewards(Q_value.detach().numpy())
                # # disc_rewards.append(disc_Q_value)
                # # Q_mean = np.mean(disc_Q_value)
                # # Q_std = np.std(disc_Q_value)
                # # for i in range(batch_size):
                # #     if Q_std != 0:
                # #         disc_Q_value[i] = (disc_Q_value[i] - Q_mean) / Q_std
                # #     else:
                # #         disc_Q_value[i] = disc_Q_value[i] - Q_mean



                # # values = torch.FloatTensor(values)
                # # Qvals = torch.FloatTensor(Qvals)
                # # advantage = Qvals - values
                # # log_prob = torch.Tensor(actions)
                # # actor_loss = (-log_prob * advantage).mean()
                # # critic_loss = 0.5 * advantage.pow(2).mean()

                # # self.critics.zero_grad()
                # # ac_loss.backward()
                # # ac_optimizer.step()




                # # self.critics[j].zero_grad()
                # # Qpred = self.critics[j].forward(states).to(self.critics[j].device)
                # # Qnext = (
                # #     self.critics_next[j].forward(next_states).to(self.critics_next[j].device)
                # # )

                # maxA = Qnext.max(1)[1]  # to(self.Q_eval.device)
                # rewards = torch.tensor(rewards, dtype=torch.float32).to(
                #     self.critics[j].device
                # )

                # Q_target = Qpred.clone()
                # for i, Qnext_a in enumerate(maxA):
                #     if not terminated[i]:
                #         Q_target[i, 0] = rewards[
                #             i
                #         ] + self.gamma * torch.max(Qnext[i, 0])
                #     else:
                #         Q_target[i, 0] = rewards[i]

                # loss = (
                #     self.critics[j].loss(Qpred, Q_target).to(self.critics[j].device)
                # )
                # loss.backward()

                self.decay_exploration(j)
            # self.mean_reward_memory.append(disc_rewards)

        # self.memory.clear()

    def discount_rewards(self, reward):
        """ computes discounted reward """
        discounted_r = [0] * len(reward)
        running_add = 0
        for j in reversed(range(0, reward.size)):
            running_add = running_add * self.gamma + reward[j]

            discounted_r[j] = running_add
        return np.array(discounted_r)

    def decay_exploration(self, j):
        "Lower the epsilon valvue to favour greedy actions"
        if self.epsilon[j] > self.epsilon_min[j]:
            self.epsilon[j] = self.epsilon[j] * self.epsilon_decay[j]

    def reset(self, init_state):
        self.action_state = init_state[0]
        self.action = None
        self.action_delay_cnt = self.action_delay

    def save_trained_model(self):
        "Save the model given a better model has been fitted"
        for i in range(self.n_tanks):
            if self.save_model[i]:
                actor_model_name = "Actor_Network_" + str(self.hl_size[i]) + "HL" + str(i)
                critic_model_name = "Critic_Network_" + str(self.hl_size[i]) + "HL" + str(i)

                actor_path = self.save_model_path + actor_model_name + ".pt"
                critic_path = self.save_model_path + critic_model_name + ".pt"
                torch.save(self.actors[i].state_dict(), actor_path)
                torch.save(self.critics[i].state_dict(), critic_path)
        print("ANN_Model was saved")
