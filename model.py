import math
import os
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init=0.5, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.bias = bias

        self.mu_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_weight', torch.Tensor(out_features, in_features))

        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('epsilon_bias', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        sqrt_in_features = math.sqrt(self.in_features)
        sqrt_out_features = math.sqrt(self.out_features)

        mu_bound = 1 / sqrt_in_features
        self.mu_weights.data.uniform_(-mu_bound, mu_bound)
        self.sigma_weights.data.fill_(self.sigma_init / sqrt_in_features)

        if self.bias:
            self.mu_bias.data.uniform_(-mu_bound, mu_bound)
            self.sigma_bias.data.fill_(self.sigma_init / sqrt_out_features)

    def reset_noise(self):
        in_epsilon = self.generate_noise(self.in_features)
        out_epsilon = self.generate_noise(self.out_features)

        self.epsilon_weight.copy_(out_epsilon.outer(in_epsilon))

        if self.bias:
            self.epsilon_bias.copy_(out_epsilon)

    @staticmethod
    def generate_noise(features: int) -> torch.Tensor:
        noise = torch.randn(features)
        return noise.sign().mul(noise.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.mu_weights + self.sigma_weights * self.epsilon_weight,
            self.mu_bias + self.sigma_bias * self.epsilon_bias if self.bias else None
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias
        )


class AbstractNet(nn.Module):
    def __init__(self, layer_class: Type, net_structure: tuple[int, ...]):
        super().__init__()
        self.layers = nn.ModuleList([
            layer_class(net_structure[layer], net_structure[layer + 1])
            for layer in range(len(net_structure) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out

    def save_checkpoint(self, path, index=None):
        model_path = os.path.join(path, f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth')
        torch.save(self.state_dict(), model_path)

    def load_checkpoint(self, path, index=None):
        model_path = os.path.join(path, f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth')
        self.load_state_dict(torch.load(model_path, map_location=device))


class Net(AbstractNet):
    def __init__(self, net_structure: tuple[int, ...]):
        super().__init__(nn.Linear, net_structure)


class NoisyNet(AbstractNet):
    def __init__(self, net_structure: tuple[int, ...]):
        super().__init__(NoisyLinear, net_structure)

    def reset_noise(self):
        for layer in self.layers:
            layer.reset_noise()


def fan_in_init(size, fan_in=None):
    fan_in = fan_in or size[0]
    v = 1. / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNet(nn.Module):
    def __init__(self, net_structure: tuple[int, ...], algorithm, init_w=None):
        super().__init__()
        self.algorithm = algorithm

        if 'ppo' in self.algorithm:
            self.layers = nn.ModuleList([
                nn.Linear(net_structure[layer], net_structure[layer + 1])
                for layer in range(len(net_structure) - 1)
            ])
        elif self.algorithm == 'wolp':
            self.layers = nn.ModuleList([
                nn.Linear(net_structure[layer], net_structure[layer + 1])
                for layer in range(len(net_structure) - 1)
            ])

            self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = fan_in_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)

    def clip_gradients(self, clip_val):
        for param in self.parameters():
            param.register_hook(lambda grad: grad.clamp_(-clip_val, clip_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, net_structure: tuple[int, ...], algorithm, init_w=None, action_dim=1):
        super().__init__()
        self.algorithm = algorithm

        if 'ppo' in self.algorithm:
            self.layers = nn.ModuleList([
                nn.Linear(net_structure[layer], net_structure[layer + 1])
                for layer in range(len(net_structure) - 1)
            ])
        elif self.algorithm == 'wolp':
            self.layers = nn.ModuleList([
                nn.Linear(net_structure[layer] + (action_dim * layer == 1), net_structure[layer + 1])
                for layer in range(len(net_structure) - 1)
            ])
            self.init_weights(init_w)

    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = fan_in_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)

    def forward(self, xs) -> torch.Tensor:
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
                    x = F.relu(layer(torch.cat([x, a], len(a.shape) - 1)))
                else:
                    x = F.relu(layer(x))
            out = self.layers[-1](x)
            return out

    def save_checkpoint(self, path, index=None):
        torch.save(self.state_dict(), os.path.join(path, f'critic_torch_{self.algorithm}_{index}.pth' if index is not None else f'critic_torch_{self.algorithm}.pth'))

    def load_checkpoint(self, path, index=None):
        self.load_state_dict(torch.load(os.path.join(path, f'critic_torch_{self.algorithm}_{index}.pth' if index is not None else f'critic_torch_{self.algorithm}.pth'), map_location=device))
