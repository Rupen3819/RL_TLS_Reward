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
    def __init__(self, network_type, algorithm_name):
        super().__init__()
        self.network_type = network_type
        self.algorithm_name = algorithm_name

    def save_checkpoint(self, path, index=None):
        model_path = self._get_model_path(path, index)
        torch.save(self.state_dict(), model_path)

    def load_checkpoint(self, path, index=None):
        model_path = self._get_model_path(path, index)
        self.load_state_dict(torch.load(model_path, map_location=device))

    def _get_model_path(self, path, index):
        model_name = f'{self.network_type}_torch_{self.algorithm_name}_{index}.pth'

        if index is not None:
            model_name += f'_{index}'

        model_name += '.pth'

        return os.path.join(path, model_name)


class ReluNet(AbstractNet):
    def __init__(self, network_type: str, algorithm_name: str, layer_class: Type, net_structure: tuple[int, ...]):
        super().__init__(network_type, algorithm_name)
        self.layers = nn.ModuleList([
            layer_class(net_structure[layer], net_structure[layer + 1])
            for layer in range(len(net_structure) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out


class QNet(ReluNet):
    def __init__(self, algorithm_name: str, net_structure: tuple[int, ...], noisy: bool = False):
        self.noisy = noisy
        network_type = 'noisy_q_network' if self.noisy else 'q_network'
        layer_class = NoisyLinear if self.noisy else nn.Linear
        super().__init__(network_type, algorithm_name, layer_class, net_structure)

    def reset_noise(self):
        if not self.noisy:
            return

        for noisy_layer in self.layers:
            noisy_layer.reset_noise()


class DuelingQNet(AbstractNet):
    def __init__(self, algorithm_name: str, net_structure: tuple[int, ...], noisy: bool = False):
        if len(net_structure) < 4:
            raise ValueError('Network structure must have at least 4 dimensions (input, two hidden dims, and output)')

        self.noisy = noisy
        network_type = 'noisy_q_network' if self.noisy else 'q_network'
        layer_class = NoisyLinear if self.noisy else nn.Linear

        super().__init__(network_type, algorithm_name)

        state_size, *hidden_dims, action_size = net_structure

        common_layers = [layer_class(state_size, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            common_layers += [
                layer_class(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU()
            ]
        self.common_stream = nn.Sequential(*common_layers)

        self.value_stream = nn.Sequential(
            layer_class(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            layer_class(hidden_dims[-1], 1)
        )

        self.advantage_stream = nn.Sequential(
            layer_class(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            layer_class(hidden_dims[-1], action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.common_stream(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + advantage - advantage.mean()

    def reset_noise(self):
        if not self.noisy:
            return

        for stream in [self.common_stream, self.value_stream, self.advantage_stream]:
            for i in range(0, len(stream), 2):
                stream[i].reset_noise()


class DistNet(nn.Module):
    def __init__(self, network_class: Type, support: torch.Tensor, algorithm_name: str, net_structure: tuple[int, ...], noisy: bool = False):
        super().__init__()
        self.support = support
        self.n_atoms = support.size(dim=0)

        state_size, *layers, action_size = net_structure
        net_structure = (state_size, *layers, action_size * self.n_atoms)
        self.action_size = action_size

        self.network = network_class(algorithm_name, net_structure, noisy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        q_atoms = self.network(x).view(-1, self.action_size, self.n_atoms)
        dist = F.softmax(q_atoms, dim=-1)
        return dist.clamp(min=1e-3)

    def reset_noise(self):
        self.network.reset_noise()

    def parameters(self):
        return self.network.parameters()

    def to(self, *args, **kwargs):
        self.network = self.network.to(*args, **kwargs)
        return self


class PpoActorNet(ReluNet):
    def __init__(self, algorithm_name, net_structure: tuple[int, ...]):
        super().__init__('actor', algorithm_name, nn.Linear, net_structure)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        out = F.softmax(x, dim=-1)
        dist = Categorical(out)
        return dist


class PpoCriticNet(ReluNet):
    def __init__(self, algorithm_name, net_structure: tuple[int, ...]):
        super().__init__('critic', algorithm_name, nn.Linear, net_structure)


class AbstractWolpNet(AbstractNet):
    def __init__(self, network_type):
        super().__init__(network_type, 'wolp')

    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = self.fan_in_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)

    def fan_in_init(size, fan_in=None):
        fan_in = fan_in or size[0]
        v = 1. / np.sqrt(fan_in)
        return torch.Tensor(size).uniform_(-v, v)


class WolpActorNet(AbstractWolpNet):
    def __init__(self, net_structure: tuple[int, ...], init_w=None):
        super().__init__('actor')
        self.layers = nn.ModuleList([
            nn.Linear(net_structure[layer], net_structure[layer + 1])
            for layer in range(len(net_structure) - 1)
        ])

        self.init_weights(init_w)

    def clip_gradients(self, clip_val):
        for param in self.parameters():
            param.register_hook(lambda grad: grad.clamp_(-clip_val, clip_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = torch.tanh(self.layers[-1](x))
        return out


class WolpCriticNet(AbstractWolpNet):
    def __init__(self, net_structure: tuple[int, ...], init_w=None, action_dim=1):
        super().__init__('critic')

        self.layers = nn.ModuleList([
            nn.Linear(net_structure[layer] + (action_dim * layer == 1), net_structure[layer + 1])
            for layer in range(len(net_structure) - 1)
        ])
        self.init_weights(init_w)

    def forward(self, xs) -> torch.Tensor:
        x, a = xs
        for layer in self.layers[:-1]:
            if layer == self.layers[1]:
                x = F.relu(layer(torch.cat([x, a], len(a.shape) - 1)))
            else:
                x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out
