import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class modularQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_dim, output_dim, hidden_dim,seed):
        super(modularQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(int(current_dim), int(hdim)))
            current_dim = hdim
        self.layers.append(nn.Linear(int(current_dim), output_dim))

    def forward(self, x):
        print(x.size())
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            print(x.size())
        out = F.relu(self.layers[-1](x))
        print(out.size())
        return out

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Net(nn.Module):
    def __init__(self, net_structure):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([(nn.Linear(int(net_structure[layer]),
                                                int(net_structure[layer+1]))) for layer in range(len(net_structure)-1)])        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out

    def save_checkpoint(self, path, index=None):
        torch.save(self.state_dict(), os.path.join(path, f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth'))

    def load_checkpoint(self, path, index=None):
        self.load_state_dict(torch.load(os.path.join(path, f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth'), map_location=device))

class ActorNet(nn.Module):
    def __init__(self, net_structure, algorithm, init_w=None):
        super(ActorNet, self).__init__()
        self.algorithm = algorithm

        if 'ppo' in self.algorithm:
            self.layers = nn.ModuleList([(nn.Linear(int(net_structure[layer]),
                                                    int(net_structure[layer+1]))) for layer in range(len(net_structure)-1)])
        elif self.algorithm == 'wolp':
            self.layers = nn.ModuleList([(nn.Linear(int(net_structure[layer]),
                                                    int(net_structure[layer+1]))) for layer in range(len(net_structure)-1)])

            self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = fanin_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)

    def clip_gradients(self, clip_val):
        for param in self.parameters():
            param.register_hook(lambda grad: grad.clamp_(-clip_val, clip_val))

    def forward(self, x):
        if 'ppo' in self.algorithm:
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            x = self.layers[-1](x)
            out = F.softmax(x, dim=-1)
            dist = Categorical(out)
            return dist
        elif self.algorithm == 'wolp':
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            out = torch.tanh(self.layers[-1](x))
            return out

    def save_checkpoint(self, path, index=None):
        torch.save(self.state_dict(), os.path.join(path, f'actor_torch_{self.algorithm}_{index}.pth' if index is not None else f'actor_torch_{self.algorithm}.pth'))

    def load_checkpoint(self, path, index=None):
        self.load_state_dict(torch.load(os.path.join(path, f'actor_torch_{self.algorithm}_{index}.pth' if index is not None else f'actor_torch_{self.algorithm}.pth'), map_location=device))

class CriticNet(nn.Module):
    def __init__(self, net_structure, algorithm, init_w=None, action_dim=1):
        super(CriticNet, self).__init__()
        self.algorithm = algorithm

        if 'ppo' in self.algorithm:
            self.layers = nn.ModuleList([(nn.Linear(int(net_structure[layer]),
                                                    int(net_structure[layer+1]))) for layer in range(len(net_structure)-1)])
        elif self.algorithm == 'wolp':
            self.layers = nn.ModuleList([(nn.Linear(int(net_structure[layer]) + action_dim if layer == 1 else int(net_structure[layer]),
                                                    int(net_structure[layer+1]))) for layer in range(len(net_structure)-1)])

            self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = fanin_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        if 'ppo' in self.algorithm:
            x = xs
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            out = self.layers[-1](x)
            return out
        elif self.algorithm == 'wolp':
            x, a = xs
            for layer in self.layers[:-1]:
                if layer == self.layers[1]:
                    x = F.relu(layer(torch.cat([x,a],len(a.shape)-1)))
                else:
                    x = F.relu(layer(x))
            out = self.layers[-1](x)
            return out

    def save_checkpoint(self, path, index=None):
        torch.save(self.state_dict(), os.path.join(path, f'critic_torch_{self.algorithm}_{index}.pth' if index is not None else f'critic_torch_{self.algorithm}.pth'))

    def load_checkpoint(self, path, index=None):
        self.load_state_dict(torch.load(os.path.join(path, f'critic_torch_{self.algorithm}_{index}.pth' if index is not None else f'critic_torch_{self.algorithm}.pth'), map_location=device))
