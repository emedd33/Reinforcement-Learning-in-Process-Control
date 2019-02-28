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
        self.save_model_bool = AGENT_PARAMS["SAVE_MODEL"]
        self.train_model = AGENT_PARAMS["TRAIN_MODEL"]
        self.model_name = AGENT_PARAMS['MODEL_NAME']

        self.state_size = AGENT_PARAMS["OBSERVATIONS"]
        self.action_state = None
        self.action_size = AGENT_PARAMS["VALVE_POSITIONS"]
        self.action_choices = self._build_action_choices(self.action_size)
        self.action = None
        self.action_delay = AGENT_PARAMS["ACTION_DELAY"]
        self.action_delay_cnt = self.action_delay

        if self.train_model:
            self.epsilon = AGENT_PARAMS["EPSILON"]
        else:
            self.epsilon = 0
        self.epsilon_min = AGENT_PARAMS["EPSILON_MIN"]
        self.epsilon_decay = AGENT_PARAMS["EPSILON_DECAY"]
        self.gamma = AGENT_PARAMS["GAMMA"]
        self.buffer = 0
        self.buffer_thres = AGENT_PARAMS["BUFFER_THRESH"]

        self.learning_rate = AGENT_PARAMS["LEARNING_RATE"]
        self.hl_size = AGENT_PARAMS["HIDDEN_LAYER_SIZE"]
        self.batch_size = AGENT_PARAMS["BATCH_SIZE"]

        self.Q_eval, self.Q_next = self._build_ANN(
            self.state_size,
            self.hl_size,
            self.action_size,
            learning_rate=self.learning_rate,
        )

    def _build_action_choices(self, action_size):
        "Create a list of the valve positions ranging from 0-1"
        valve_positions = []
        for i in range(action_size):
            valve_positions.append((i) / (action_size - 1))
        return np.array(list(reversed(valve_positions)))

    def _build_ANN(self, input_size, hidden_size, action_size, learning_rate):
        if self.load_model:
            Q_net = Net(input_size, hidden_size, action_size, learning_rate)
            model_name = self.model_name
            path = "Q_learning/Tank_1/saved_networks/" + model_name + ".pt"
            Q_net.load_state_dict(torch.load(path))
            Q_net.eval()
            return Q_net, Q_net
        "Creates or loads a ANN valve function approximator"

        Q_eval = Net(input_size, hidden_size, action_size, learning_rate)
        Q_next = Net(input_size, hidden_size, action_size, learning_rate)
        return Q_eval, Q_next

    def get_z(self, action):
        self.action = action
        z = self.action_choices[action]
        return z

    def remember(self, state, reward, done, t):
        "Stores instances of each time step"
        if self.train_model:
            if done:
                if len(state) <= self.action_delay+2:
                    action_state = state[0]
                else:
                    action_state_index = -self.action_delay_cnt-2
                    action_state = state[action_state_index]
                self.memory.append(
                    np.array(
                        [action_state, self.action, reward, state[-1], done]
                    )
                )
                self.action_delay_cnt = -1
                self.buffer += 1
            elif (
                self.action_delay_cnt >= self.action_delay
                and t >= self.action_delay
            ):
                action_state = state[-self.action_delay-2]
                self.memory.append(
                    np.array(
                        [action_state, self.action, reward, state[-1], done]
                    )
                )
                self.action_delay_cnt = -1
                self.buffer += 1

    def act_greedy(self, state):
        "Predict the optimal action to take given the current state"

        choice = self.Q_eval.forward(state)
        action = torch.argmax(choice).item()
        return action

    def act(self, state):
        """
        Agent uses the state and gives either an
        action of exploration or explotation
        """
        if self.action_delay_cnt >= self.action_delay:
            if np.random.rand() <= self.epsilon:  # Exploration
                random_action = random.randint(0, self.action_size - 1)
                self.action = random_action
                return self.action
            self.action = self.act_greedy(state)  # Exploitation
            return self.action
        else:
            self.action_delay_cnt += 1
            return self.action

    def is_ready(self):
        "Check if enough data has been collected"
        if not self.train_model:  # Model has been set to not collect data
            return False
        if len(self.memory) < self.batch_size:
            return False
        if self.buffer < self.buffer_thres:
            return False
        return True

    def Qreplay(self, e):
        """"
        Train the model to improve the predicted value of consecutive
        recurring states, Off policy Q-learning with batch training
        """
        self.Q_eval.zero_grad()
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        states = np.stack(minibatch[:, 0])
        actions = np.stack(minibatch[:, 1])
        rewards = np.stack(minibatch[:, 2])
        next_states = np.stack(minibatch[:, 3])
        terminated = np.stack(minibatch[:, 4])

        self.Q_eval.zero_grad()
        Qpred = self.Q_eval.forward(states).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(next_states).to(self.Q_next.device)

        maxA = Qnext.max(1)[1]  # to(self.Q_eval.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(
            self.Q_eval.device
        )

        Q_target = Qpred.clone()
        for i, Qnext_a in enumerate(maxA):
            if not terminated[i]:
                Q_target[i, actions[i]] = rewards[i] + self.gamma * torch.max(
                    Qnext[i, Qnext_a]
                )
            else:
                Q_target[i, actions[i]] = rewards[i]
        loss = self.Q_eval.loss(Qpred, Q_target).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()
        self.decay_exploration()

    def decay_exploration(self):
        "Lower the epsilon valvue to favour greedy actions"

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset(self, init_state):
        self.action_state = init_state[0]
        self.action = None
        self.action_delay_cnt = self.action_delay

    def save_model(self, mean_reward, max_mean_reward):
        "Save the model given a better model has been fitted"

        if mean_reward >= max_mean_reward:

            model_name = "Network_" + str(self.hl_size) + "HL" \
                # + str(int(mean_reward))
            path = "Q_learning/Tank_1/saved_networks/" + model_name + ".pt"
            torch.save(self.Q_eval.state_dict(), path)
            print("ANN_Model was saved")
            max_mean_reward = mean_reward
        return max_mean_reward
