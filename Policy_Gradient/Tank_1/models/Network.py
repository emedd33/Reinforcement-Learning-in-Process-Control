import torch
import torch.nn as nn

# from IPython.core.debugger import set_trace


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, learning_rate):
        super(Net, self).__init__()
        self.n_hl = len(hidden_size)
        if self.n_hl == 0:  # No hidden layer
            self.input = nn.Linear(input_size, action_size)
            self.hl1 = None
            self.hl2 = None
        elif self.n_hl == 1:  # One hidden layer
            self.input = nn.Linear(input_size, hidden_size[0])
            self.relu = nn.ReLU()
            self.hl1 = nn.Linear(hidden_size[0], action_size)
            self.hl2 = None
        elif self.n_hl == 2:  # Two hidden layers
            self.input = nn.Linear(input_size, hidden_size[0])
            self.relu = nn.ReLU()
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
        out = self.input(x)
        if self.n_hl == 0:
            pass
        elif self.n_hl == 1:
            # out = self.relu(out)
            out = self.hl1(out)
        elif self.n_hl == 2:
            # out = self.relu(out)
            out = self.hl1(out)
            out = self.relu(out)
            out = self.hl2(out)
        return out
