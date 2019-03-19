import torch
import torch.nn as nn
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

    def backward(self, states, actions, rewards, dummy_index):
        self.optimizer.zero_grad()
        for i in range(len(states)):
            if i == dummy_index:
                break
            state = states[i]
            # action = torch.autograd.Variable(torch.FloatTensor([actions[i]]))
            reward = rewards[i]

            probs = self.forward(state)
            m = Bernoulli(probs)
            loss = (
                -m.log_prob(actions[i]) * reward
            )  # Negtive score function x reward
            loss.backward()

            self.optimizer.step()
