import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.distributions import Bernoulli

# from IPython.core.debugger import set_trace


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, learning_rate, action_size=1):
        super(Net, self).__init__()
        self.action_size = action_size
        self.n_hl = len(hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if self.n_hl == 0:  # No hidden layer
            self.input = nn.Linear(input_size, action_size)
            self.hl1 = None
            self.hl2 = None
        elif self.n_hl == 1:  # One hidden layer
            self.input = nn.Linear(input_size, hidden_size[0])
            self.hl1 = nn.Linear(hidden_size[0], action_size)
            self.hl2 = None
        elif self.n_hl == 2:  # Two hidden layers
            self.input = nn.Linear(input_size, hidden_size[0])
            self.hl1 = nn.Linear(hidden_size[0], hidden_size[1])
            self.hl2 = nn.Linear(hidden_size[1], action_size)
        else:
            raise ValueError(
                "Not supported network with more than 2 hidden layers"
            )

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        out_init = self.input(x)
        if self.n_hl == 0:
            out_final = out_init
        elif self.n_hl == 1:
            # out = self.relu(out)
            out_final = self.hl1(out_init)
        elif self.n_hl == 2:
            # out = self.relu(out)
            out_1 = self.hl1(out_init)
            out_2 = self.relu(out_1)
            out_final = self.hl2(out_2)
        out = self.sigmoid(out_final)
        return out

    def actor_backward(self, values, actions, q_values, dummy_index):
        q_values = torch.Tensor(q_values)
        actions = torch.Tensor(actions)

        advantage = q_values - values
        actor_loss = (-actions * advantage).mean()

        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer.step()
        # for i in range(len(states)):
        #     state = states[i]
        #     # action = torch.autograd.Variable(torch.FloatTensor([actions[i]]))
        #     q_value = q_values[i]

        #     probs = self.forward(state)
        #     m = Bernoulli(probs)
        #     loss = (
        #         -m.log_prob(actions[i]) * q_value
        #     )  # Negtive score function x reward
        #     loss.backward()

        #     self.optimizer.step()

    def critic_backward(self, values, actions, q_values, dummy_index):
        q_values = torch.Tensor(q_values)
        actions = torch.Tensor(actions)

        advantage = q_values - values
        critic_loss = 0.5 * advantage.pow(2).mean()

        self.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.optimizer.step()
