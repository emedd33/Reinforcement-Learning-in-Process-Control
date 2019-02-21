import torch
import torch.nn as nn

# from IPython.core.debugger import set_trace


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, learning_rate):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.out = nn.Linear(hidden_size[0], action_size)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        out = self.out(out)
        return out
