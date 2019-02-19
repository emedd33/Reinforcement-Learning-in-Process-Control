import torch
from torch.autograd import Variable
from IPython.core.debugger import set_trace
import torch.nn as nn
import numpy as np

class NetWork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, action_size,learning_rate):
        super(NetWork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0]) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], action_size)  
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = torch.Tensor(state).to(self.device)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    