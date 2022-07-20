import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNet, DuelingQNet
from memory.replay import ReplayBuffer, NStepReplayBuffer, PrioritizedReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RainbowDQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self, state_size: int, action_size: int, hidden_dim: tuple[int, ...], fixed_action_space: bool,
            traffic_lights: dict[str, str], buffer_size: int = int(1e5), batch_size: int = 64, gamma: float = 0.99,
            tau: float = 1e-3, learning_rate: float = 5e-4, update_interval: int = 4,
            double_q_learning: bool = True, n_step_bootstrapping: int = 1, noisy_net: bool = True,
            dueling_net: bool = True, prioritized_replay: bool = True, alpha: float = 1, beta: float = 1
    ):
        """Initialize a DQN Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_dim: (tuple): dimensions of hidden layers
            fixed_action_space (bool): whether action space is linear (8) or exponential (8^n)
            traffic_lights (dict): traffic light ids from the SUMO environment
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau(float): for soft update of target parameters
            learning_rate (float): learning rate
            update_interval (int): how often to update the network
            double_q_learning (bool): if double Q learning should be used to decouple action selection from evaluation
            n_step_bootstrapping (int):
            noisy_net (bool):
            dueling_net (bool):
            prioritized_replay (bool):
            alpha (float):
            beta (float):

        """
        self.state_size = state_size
        if fixed_action_space:
            self.state_size += len(traffic_lights)

        self.action_size = action_size
        self.fixed_action_space = fixed_action_space

        self.traffic_lights = traffic_lights

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_interval = update_interval

        self.double_q_learning = double_q_learning
        self.n_step_bootstrapping = n_step_bootstrapping
        self.noisy_net = noisy_net
        self.dueling_net = dueling_net
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta = beta

        # Q-Network
        print(hidden_dim)
        net_structure = (self.state_size, hidden_dim[0], hidden_dim[1], self.action_size)

        network_class = DuelingQNet if self.dueling_net else QNet

        self.local_q_network = network_class('rainbow_dqn', net_structure, noisy=self.noisy_net).to(device)
        self.target_q_network = network_class('rainbow_dqn', net_structure, noisy=self.noisy_net).to(device)
        print(self.local_q_network)

        self.optimizer = optim.Adam(self.local_q_network.parameters(), lr=learning_rate)

        # Replay memory
        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, self.batch_size, self.alpha, self.beta)
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        if self.n_step_bootstrapping > 1:
            step_buffers = len(self.traffic_lights) if fixed_action_space else 1
            self.memory = NStepReplayBuffer(self.memory, self.n_step_bootstrapping, self.gamma, step_buffers)

        # Initialize time step (for updating every update_interval steps)
        self.t_step = 0

    def one_hot_state(self, index, state):
        one_hot = np.zeros(len(self.traffic_lights))
        np.put(one_hot, index, 1)
        state_one_hot = np.array(one_hot.tolist() + state)
        return state_one_hot

    def step(self, state, action, reward, next_state, done):
        reward = sum(reward.values())

        # Save experience in replay memory
        if self.fixed_action_space:
            for index, traffic_light_id in enumerate(self.traffic_lights):
                state_one_hot = self.one_hot_state(index, state)
                next_state_one_hot = self.one_hot_state(index, next_state)
                self.memory.add(state_one_hot, action[traffic_light_id], reward, next_state_one_hot, done)
        else:
            self.memory.add(state, action, reward, next_state, done)

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
        if self.fixed_action_space:
            action_dict = dict.fromkeys(self.traffic_lights, 0)
            action_values_dict = dict.fromkeys(self.traffic_lights, 0)

            for index, traffic_light_id in enumerate(self.traffic_lights):
                state_one_hot = self.one_hot_state(index, state.tolist())
                action_values, action = self.choose_action(state_one_hot, eps)
                action_values_dict[traffic_light_id] = action_values
                action_dict[traffic_light_id] = action

            return action_values_dict, action_dict
        else:
            return self.choose_action(state, eps)

    def choose_action(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.local_q_network.eval()
        with torch.no_grad():
            action_values = self.local_q_network(state)
        self.local_q_network.train()

        # Action selection is epsilon greedy if noisy net is disabled, otherwise it is always greedy
        choose_greedily = self.noisy_net or random.random() > eps
        if choose_greedily:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action_values.tolist(), action

    def learn(self, experiences, gamma: float):
        """
        Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.compute_next_q_targets(next_states)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.local_q_network(states).gather(1, actions)

        # Compute loss
        if self.prioritized_replay:
            new_priorities = q_targets - q_expected
            self.memory.update_priorities(new_priorities)
            loss = torch.mean((weights * new_priorities) ** 2)
        else:
            loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.tau)

        if self.noisy_net:
            self.local_q_network.reset_noise()
            self.target_q_network.reset_noise()

    def compute_next_q_targets(self, next_states):
        if self.double_q_learning:
            local_argmax_states = self.local_q_network(next_states).detach().argmax(1)
            return self.target_q_network(next_states).detach()[
                torch.arange(self.batch_size), local_argmax_states
            ].unsqueeze(1)
        else:
            return self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)

    def soft_update(self, tau: float):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            tau (float): interpolation parameter
        """
        # Copy weights from local model to target model, using tau as interpolation factor
        for target_param, local_param in zip(self.target_q_network.parameters(), self.local_q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, path):
        print('Saving model')
        self.local_q_network.save_checkpoint(path)
        print('Model saved successfully')

    def load_model(self, path):
        print('Loading model')
        self.local_q_network.load_checkpoint(path)
        print('Model loaded sucessfully')