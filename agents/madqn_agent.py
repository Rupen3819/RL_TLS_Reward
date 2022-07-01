import numpy as np
import random

from model import Net
from agents.memory import MultiSequentialMemory

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self, state_size: int, action_size: int, num_agents: int, hidden_dim: tuple[int, ...], single_state_space: bool, local_reward_signal: bool,
            traffic_lights: dict[str, str], buffer_size: int = int(1e5), batch_size: int = 64, gamma: float = 0.99,
            tau: float = 1e-3, learning_rate: float = 5e-4, update_interval: int = 4
    ):
        """Initialize MADQN Agent objects.

        Params
        ======

            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in multi-agent implementation
            hidden_dim: (tuple): dimensions of hidden layers
            single_state_space (bool): whether state space for each agent is single (17) or linear (17*n)
            local_reward_signal (bool): whether reward signal for each agent is local (r) or global (sum(r))
            traffic_lights (dict): traffic light ids from the SUMO environment
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau(float): for soft update of target parameters
            learning_rate (float): learning rate
            update_interval (int): how often to update the network
        """
        self.state_size = state_size
        if single_state_space:
            self.state_size //= num_agents

        self.action_size = action_size
        self.num_agents = num_agents
        self.single_state_space = single_state_space
        self.local_reward_signal = local_reward_signal

        self.traffic_lights = traffic_lights

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_interval = update_interval

        # Q-Networks
        print(hidden_dim)
        hidden_dim = (self.state_size, hidden_dim[0], hidden_dim[1], self.action_size)

        self.local_q_networks = [Net(hidden_dim).to(device) for _ in range(self.num_agents)]
        self.target_q_networks = [Net(hidden_dim).to(device) for _ in range(self.num_agents)]
        print(self.local_q_networks)

        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.local_q_networks]

        # Replay memory
        self.memory = MultiSequentialMemory(self.buffer_size, self.batch_size, self.single_state_space, self.local_reward_signal)

        # Initialize time step (for updating every update_interval steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.single_state_space:
            states = [state[i:i + self.state_size] for i in range(0, len(state), self.state_size)]
            next_states = [next_state[i:i + self.state_size] for i in range(0, len(next_state), self.state_size)]
        else:
            states = state
            next_states = next_state

        if self.local_reward_signal:
            rewards = list(reward.values())
        else:
            rewards = sum(list(reward.values()))

        actions = list(action.values())

        self.memory.add(states, actions, rewards, next_states, done)

        # Learn every update_interval time steps.
        self.t_step += 1
        if self.t_step % self.update_interval == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if self.single_state_space:
            state = torch.from_numpy(state).float().unsqueeze(0).view(self.num_agents, -1).to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        actions = dict.fromkeys(self.traffic_lights, 0)
        actions_values = dict.fromkeys(self.traffic_lights, 0)

        for agent_id in range(self.num_agents):
            self.local_q_networks[agent_id].eval()

            with torch.no_grad():
                if self.single_state_space:
                    action_values = self.local_q_networks[agent_id](state[agent_id, :])
                else:
                    action_values = self.local_q_networks[agent_id](state)

            self.local_q_networks[agent_id].train()

            actions_values[f'TL{agent_id + 1}'] = action_values.tolist()

            # Epsilon-greedy action selection
            if random.random() > eps:
                actions[f'TL{agent_id + 1}'] = np.argmax(action_values.cpu().data.numpy())
            else:
                actions[f'TL{agent_id + 1}'] = random.choice(np.arange(self.action_size))

        return actions_values, actions

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        for agent_id in range(self.num_agents):
            # Get max predicted Q values (for next states) from target model
            if self.single_state_space:
                q_targets_next = self.target_q_networks[agent_id](next_states[:, agent_id]).detach().max(1)[0].unsqueeze(1)
            else:
                q_targets_next = self.target_q_networks[agent_id](next_states).detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states
            if self.local_reward_signal:
                q_targets = rewards[:, agent_id].view(-1, 1) + (gamma * q_targets_next * (1 - dones))
            else:
                q_targets = rewards + (gamma * q_targets_next * (1 - dones))

            # Get expected Q values from local model
            if self.single_state_space:
                q_expected = self.local_q_networks[agent_id](states[:, agent_id]).gather(1, actions[:, agent_id].view(-1, 1))
            else:
                q_expected = self.local_q_networks[agent_id](states).gather(1, actions[:, agent_id].view(-1, 1))

            # Compute loss
            loss = F.mse_loss(q_expected, q_targets)

            # Minimize the loss
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()

            # Update target network
            self.soft_update(self.local_q_networks[agent_id], self.target_q_networks[agent_id], self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_models(self, path):
        print('... saving models ...')
        for agent_id, local_q_network in enumerate(self.local_q_networks):
            local_q_network.save_checkpoint(path, str(agent_id))
        print('... models saved successfully ...')

    def load_models(self, path):
        print('... loading models ...')
        for agent_id, local_q_network in enumerate(self.local_q_networks):
            local_q_network.load_checkpoint(path, str(agent_id))
        print('... models loaded successfully ...')
