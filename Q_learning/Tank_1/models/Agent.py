from collections import deque
import torch
from .Network import Net
import numpy as np
import random


class Agent:
    def __init__(self, AGENT_PARAMS):
        "Parameters are set in the params.py file"
        self.memory_size = AGENT_PARAMS["MEMORY_LENGTH"]
        self.memory = deque(maxlen=self.memory_size)
        self.load_model = AGENT_PARAMS["LOAD_MODEL"]
        self.model_name = AGENT_PARAMS["LOAD_MODEL_NAME"]
        self.save_model = AGENT_PARAMS["SAVE_MODEL"]
        self.train_model = AGENT_PARAMS["TRAIN_MODEL"]

        self.load_model_path = AGENT_PARAMS["LOAD_MODEL_PATH"]
        self.save_model_path = AGENT_PARAMS["SAVE_MODEL_PATH"]

        self.n_tanks = AGENT_PARAMS["N_TANKS"]
        self.state_size = AGENT_PARAMS["OBSERVATIONS"]
        self.action_state = None
        self.action_size = AGENT_PARAMS["VALVE_POSITIONS"]
        self.action_choices = self._build_action_choices(self.action_size)
        self.actions = None
        self.action_delay_cnt = [9] * self.n_tanks
        self.action_delay = AGENT_PARAMS["ACTION_DELAY"]

        self.epsilon = AGENT_PARAMS["EPSILON"]
        self.epsilon_min = AGENT_PARAMS["EPSILON_MIN"]
        self.epsilon_decay = AGENT_PARAMS["EPSILON_DECAY"]
        self.gamma = AGENT_PARAMS["GAMMA"]

        self.learning_rate = AGENT_PARAMS["LEARNING_RATE"]
        self.hl_size = AGENT_PARAMS["HIDDEN_LAYER_SIZE"]
        self.batch_size = AGENT_PARAMS["BATCH_SIZE"]

        self.Q_eval, self.Q_next = [], []
        for i in range(self.n_tanks):
            Q_eval_, Q_next_ = self._build_ANN(
                self.state_size,
                self.hl_size,
                self.action_size,
                learning_rate=self.learning_rate,
                i=i,
            )
            self.Q_eval.append(Q_eval_)
            self.Q_next.append(Q_next_)

    def _build_action_choices(self, action_size):
        "Create a list of the valve positions ranging from 0-1"
        valve_positions = []
        for i in range(action_size):
            valve_positions.append((i) / (action_size - 1))
        return np.array(list(reversed(valve_positions)))

    def _build_ANN(
        self, input_size, hidden_size, action_size, learning_rate, i
    ):
        if self.load_model[i]:
            path = (
                self.load_model_path
                + self.model_name[i]
                + ".pt"
            )
            pytorch_path = torch.load(path)
            n_hl = (len(pytorch_path)-3)
            if n_hl == 0:  # zero hidden later
                h_size = []
            elif n_hl == 1:  # 1 hidden layer
                h_size = [len(pytorch_path['input.weight'])]
            elif n_hl == 3:  # 2 hidden layers
                h_size = [len(pytorch_path['input.weight']), len(pytorch_path['hl1.bias'])]
            else:
                raise ValueError
            Q_net = Net(input_size, h_size, action_size, learning_rate[i])
            Q_net.load_state_dict(pytorch_path)
            Q_net.eval()
            return Q_net, Q_net
        "Creates or loads a ANN valve function approximator"

        Q_eval = Net(input_size, hidden_size[i], action_size, learning_rate[i])
        Q_next = Net(input_size, hidden_size[i], action_size, learning_rate[i])
        return Q_eval, Q_next

    def get_z(self, action):
        z = []
        for action in self.actions:
            z.append(self.action_choices[action])
        return z

    def remember(self, states, reward, terminated, t):
        "Stores instances of each time step"

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

    def act_greedy(self, state, i):
        "Predict the optimal action to take given the current state"

        choice = self.Q_eval[i].forward(state[i])
        action = torch.argmax(choice).item()
        return action

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
                    random_action = random.randint(0, self.action_size - 1)
                    action = random_action
                    actions.append(action)
                else:
                    action = self.act_greedy(state, i)  # Exploitation
                    actions.append(action)
            else:
                actions.append(self.actions[i])
                self.action_delay_cnt[i] += 1
        self.actions = actions
        return self.actions

    def is_ready(self):
        "Check if enough data has been collected"
        if len(self.memory) < self.batch_size:
            return False
        return True

    def Qreplay(self, e):
        """"
        Train the model to improve the predicted value of consecutive
        recurring states, Off policy Q-learning with batch training
        """
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        for j in range(self.n_tanks):
            if self.train_model[j]:
                agent_batch = minibatch[:, j]
                dummy_data = np.stack(agent_batch[:, 5])
                dummy_data_index = np.where(dummy_data)[0]
                agent_batch_comp = np.delete(agent_batch, dummy_data_index, axis=0)

                states = np.stack(agent_batch_comp[:, 0])
                actions = np.stack(agent_batch_comp[:, 1])
                rewards = np.stack(agent_batch_comp[:, 2])
                next_states = np.stack(agent_batch_comp[:, 3])
                terminated = np.stack(agent_batch_comp[:, 4])

                self.Q_eval[j].zero_grad()
                Qpred = self.Q_eval[j].forward(states).to(self.Q_eval[j].device)
                Qnext = (
                    self.Q_next[j].forward(next_states).to(self.Q_next[j].device)
                )

                maxA = Qnext.max(1)[1]  # to(self.Q_eval.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(
                    self.Q_eval[j].device
                )

                Q_target = Qpred.clone()
                for i, Qnext_a in enumerate(maxA):
                    if not terminated[i]:
                        Q_target[i, actions[i]] = rewards[
                            i
                        ] + self.gamma * torch.max(Qnext[i, Qnext_a])
                    else:
                        Q_target[i, actions[i]] = rewards[i]
                loss = (
                    self.Q_eval[j].loss(Qpred, Q_target).to(self.Q_eval[j].device)
                )
                loss.backward()

                self.Q_eval[j].optimizer.step()
                self.decay_exploration(j)

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
                model_name = "Network_" + str(self.hl_size[i]) + "HL" + str(i)

                path = self.save_model_path + model_name + ".pt"
                torch.save(self.Q_eval[i].state_dict(), path)
        print("ANN_Model was saved")
