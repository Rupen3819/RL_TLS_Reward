import numpy as np
import random

from model import Net
from agents.memory import MultiSequentialMemory

import torch
import torch.nn.functional as F
import torch.optim as optim

#BUFFER_SIZE = int(1e5)  # replay buffer size
#BATCH_SIZE = 64         # minibatch size
#GAMMA = 0.99            # discount factor
#TAU = 1e-3              # for soft update of target parameters
#LR = 5e-4               # learning rate 
#UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, hidden_dim, single_state_space, local_reward_signal, TL_list, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY):
        """Initialize MADQN Agent objects.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in multi-agent implementation
            single_state_space (bool): whether state space for each agent is single (17) or linear (17*n)
            local_reward_signal (bool): whether reward signal for each agent is local (r) or global (sum(r))
            TL_list (dict): traffic light ids from the SUMO environment
        """
        self.state_size = int(state_size / num_agents) if single_state_space else state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.single_state_space = single_state_space
        self.local_reward_signal = local_reward_signal

        self.TL_list = TL_list
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR = LR
        self.UPDATE_EVERY = UPDATE_EVERY


        # Q-Networks
        print(hidden_dim)
        hidden_dim.insert(0, self.state_size)
        hidden_dim.append(self.action_size)

        self.qnetworks_local = [Net(hidden_dim).to(device)] * self.num_agents
        print(self.qnetworks_local)
        self.qnetworks_target = [Net(hidden_dim).to(device)] * self.num_agents

        self.optimizers = [optim.Adam(n.parameters(), lr=LR) for n in self.qnetworks_local]

        # Replay memory
        self.memory = MultiSequentialMemory(self.BUFFER_SIZE, self.BATCH_SIZE, self.single_state_space, self.local_reward_signal)

        # Initialize time step (for updating every UPDATE_EVERY steps)
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

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if self.single_state_space:
            state = torch.from_numpy(state).float().unsqueeze(0).view(self.num_agents, -1).to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        actions = dict.fromkeys(self.TL_list, 0)
        actions_values = dict.fromkeys(self.TL_list, 0)

        for agent_id in range(self.num_agents):
            self.qnetworks_local[agent_id].eval()
            with torch.no_grad():
                if self.single_state_space:
                    action_values = self.qnetworks_local[agent_id](state[agent_id, :])
                else:
                    action_values = self.qnetworks_local[agent_id](state)
            self.qnetworks_local[agent_id].train()

            actions_values[f'TL{agent_id + 1}'] = action_values.tolist()

            # Epsilon-greedy action selection
            if random.random() > eps:
                actions[f'TL{agent_id + 1}'] = np.argmax(action_values.cpu().data.numpy())
            else:
                actions[f'TL{agent_id + 1}'] = random.choice(np.arange(self.action_size))

        return actions_values, actions

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        for agent_id in range(self.num_agents):
            # Get max predicted Q values (for next states) from target model
            if self.single_state_space:
                Q_targets_next = self.qnetworks_target[agent_id](next_states[:, agent_id]).detach().max(1)[0].unsqueeze(1)
            else:
                Q_targets_next = self.qnetworks_target[agent_id](next_states).detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states
            if self.local_reward_signal:
                Q_targets = rewards[:, agent_id].view(-1, 1) + (gamma * Q_targets_next * (1 - dones))
            else:
                Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            if self.single_state_space:
                Q_expected = self.qnetworks_local[agent_id](states[:, agent_id]).gather(1, actions[:, agent_id].view(-1, 1))
            else:
                Q_expected = self.qnetworks_local[agent_id](states).gather(1, actions[:, agent_id].view(-1, 1))

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetworks_local[agent_id], self.qnetworks_target[agent_id], self.TAU)

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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_models(self, path):
        print('... saving models ...')
        for agent_id in range(self.num_agents):
            self.qnetworks_local[agent_id].save_checkpoint(path, f'{agent_id}')

    def load_models(self, path):
        print('... loading models ...')
        for agent_id in range(self.num_agents):
            self.qnetworks_local[agent_id].load_checkpoint(path, f'{agent_id}')
