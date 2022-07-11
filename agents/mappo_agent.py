from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim

from environment import get_intersection_name
from model import PpoActorNet, PpoCriticNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAPPOAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self, state_size: int, action_size: int, num_agents: int, actor_dim: tuple[int, ...],
            critic_dim: tuple[int, ...], training_strategy: str, actor_parameter_sharing: bool,
            critic_parameter_sharing: bool, single_state_space: bool, local_reward_signal: bool,
            traffic_lights: dict[str, str], batch_size: int = 256, n_epochs: int = 10, policy_clip: int = 0.1,
            gamma: float = 0.99, gae_lambda: float = 0.95, policy_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3
    ):
        """
        Initialize MAPPO Agent objects.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in multi-agent implementation
            actor_dim
            critic_dim
            training_strategy (string):
                - nonstrategic:
                    - each agent learns its own individual policy which is independent
                    - multiple policies are optimized simultaneously
                    - actor_parameter_sharing and critic_parameter_sharing are turned off
                - concurrent:
                    - each agent learns its own individual policy which is independent
                    - multiple policies are optimized simultaneously
                - centralized:
                    - centralized training and decentralized execution
                    - decentralized actor maps it's local observations to action using individual policy
                    - centralized critic takes the state from all agents as input, each actor has its own critic for estimating
                      the value function, which allows each actor has different reward structure, e.g., cooperative, competitive, mixed task
            actor_parameter_sharing (bool):
                - True: all actors share a single policy which enables parameters and experiences sharing,
                        this is mostly useful where the agents are homogeneous
                - False: each actor use independent policy
            critic_parameter_sharing (bool):
                - True: all actors share a single critic which enables parameters and experiences sharing,
                        this is mostly useful where the agents are homogeneous and reward sharing holds
                - False: each actor use independent critic (though each critic can take other agents actions as input)
            single_state_space (bool): whether state space for each agent is single (17) or linear (17*n)
            local_reward_signal (bool): whether reward signal for each agent is local (r) or global (sum(r))
            traffic_lights (dict): traffic light ids from the SUMO environment
            batch_size (int): minibatch size
            n_epochs (int): the optimizer’s number of epochs
            policy_clip (int): clipping parameter epsilon
            gamma (float): discount factor
            gae_lambda (float): factor for trade-off of bias vs variance for Generalized Advantage Estimator
            policy_learning_rate (float): policy net learning rate
            critic_learning_rate (float): critic net learning rate
        """
        self.state_size = state_size
        if single_state_space:
            self.state_size //= num_agents

        self.action_size = action_size
        self.num_agents = num_agents
        self.training_strategy = training_strategy
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing
        self.single_state_space = single_state_space
        self.local_reward_signal = local_reward_signal

        self.traffic_lights = traffic_lights

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate

        # Actor-Critic-Networks
        print(actor_dim)
        actor_dim = (self.state_size, actor_dim[0], actor_dim[1], self.action_size)

        print(critic_dim)
        if self.training_strategy == 'centralized':
            critic_state_dim = self.state_size * self.num_agents
            critic_dim = (critic_state_dim, critic_dim[0], critic_dim[1], 1)
        else:
            critic_dim = (self.state_size, critic_dim[0], critic_dim[1], 1)

        if self.actor_parameter_sharing and not self.training_strategy == 'nonstrategic':
            self.actors = [PpoActorNet('mappo', actor_dim).to(device) for _ in range(self.num_agents)]
            self.actor_optimizers = [optim.Adam(actor.parameters(), lr=policy_learning_rate) for actor in self.actors]
        else:
            self.actors = [PpoActorNet('mappo', actor_dim).to(device)] * self.num_agents
            self.actor_optimizers = [optim.Adam(self.actors[0].parameters(), lr=policy_learning_rate)] * self.num_agents
        print(self.actors)

        if self.critic_parameter_sharing and not self.training_strategy == 'nonstrategic':
            self.critics = [PpoCriticNet('mappo', critic_dim).to(device) for _ in range(self.num_agents)]
            self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_learning_rate) for critic in self.critics]
        else:
            self.critics = [PpoCriticNet('mappo', critic_dim).to(device)] * self.num_agents
            self.critic_optimizers = [optim.Adam(self.critics[0].parameters(), lr=critic_learning_rate)] * self.num_agents
        print(self.critics)

        # Replay memory
        self.memory = MAPPOMemory(self.state_size, self.action_size, self.num_agents, self.single_state_space, self.batch_size)

    def remember(self, states, actions, probs, vals, rewards, dones):
        # Save experience in replay memory
        if self.single_state_space:
            states = [states[i:i + self.state_size] for i in range(0, len(states), self.state_size)]

        if self.local_reward_signal:
            rewards = list(rewards.values())
        else:
            rewards = sum(rewards.values())

        actions = list(actions.values())

        self.memory.store_memory(states, actions, probs, vals, rewards, dones)

    def choose_action(self, observation):
        # Returns actions, probs and values for given observation as per current policy
        if self.single_state_space:
            state = torch.tensor([observation], dtype=torch.float).view(self.num_agents, -1).to(device)
            whole_state = state.view(-1, self.state_size * self.num_agents)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(device)

        actions = dict.fromkeys(self.traffic_lights, 0)
        probs = [0] * self.num_agents
        values = [0] * self.num_agents

        for agent_id in range(self.num_agents):
            if self.single_state_space:
                dist = self.actors[agent_id](state[agent_id, :])

                if self.training_strategy == 'centralized':
                    value = self.critics[agent_id](whole_state)
                else:
                    value = self.critics[agent_id](state[agent_id, :])

            else:
                dist = self.actors[agent_id](state)
                value = self.critics[agent_id](state)

            action = dist.sample()

            probs[agent_id] = torch.squeeze(dist.log_prob(action)).item()
            actions[get_intersection_name(agent_id)] = torch.squeeze(action).item()
            values[agent_id] = torch.squeeze(value).item()

        return actions, probs, values

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, rewards, dones, whole_state_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros((len(rewards), self.num_agents), dtype=np.float32)

            # This can probably be done more efficiently
            for t in range(len(rewards) - 1):
                discount = 1
                action = 0

                for k in range(t, len(rewards) - 1):
                    action += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda

                for agent_id in range(self.num_agents):
                    advantage[t][agent_id] = action[agent_id]

            advantage = torch.tensor(advantage).to(device)
            values = torch.tensor(values).to(device)
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(device)

                if self.single_state_space:
                    whole_states = torch.tensor(whole_state_arr[batch], dtype=torch.float).to(device)

                old_probs = torch.tensor(old_prob_arr[batch]).to(device)
                actions = torch.tensor(action_arr[batch]).to(device)

                for agent_id in range(self.num_agents):
                    if self.single_state_space:
                        dist = self.actors[agent_id](states[:, agent_id])

                        if self.training_strategy == 'centralized':
                            critic_value = self.critics[agent_id](whole_states)
                        else:
                            critic_value = self.critics[agent_id](states[:, agent_id])
                    else:
                        dist = self.actors[agent_id](states)
                        critic_value = self.critics[agent_id](states)

                    critic_value = torch.squeeze(critic_value)

                    new_probs = dist.log_prob(actions[:, agent_id])
                    prob_ratio = new_probs.exp() / old_probs[:, agent_id].exp()
                    weighted_probs = advantage[batch][:, agent_id] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch][:, agent_id]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch][:, agent_id] + values[batch][:, agent_id]
                    critic_loss = (returns - critic_value) ** 2
                    critic_loss = critic_loss.mean()

                    total_loss = actor_loss + 0.5 * critic_loss
                    self.actor_optimizers[agent_id].zero_grad()
                    self.critic_optimizers[agent_id].zero_grad()
                    total_loss.backward()
                    self.actor_optimizers[agent_id].step()
                    self.critic_optimizers[agent_id].step()

        self.memory.clear_memory()
    
    def save_models(self, path):
        print('... saving models ...')
        for agent_id in range(self.num_agents):
            self.actors[agent_id].save_checkpoint(path, f'{agent_id}')
            self.critics[agent_id].save_checkpoint(path, f'{agent_id}')
        print('... models saved successfully')

    def load_models(self, path):
        print('... loading models ...')
        for agent_id in range(self.num_agents):
            self.actors[agent_id].load_checkpoint(path, f'{agent_id}')
            self.critics[agent_id].load_checkpoint(path, f'{agent_id}')
        print('... models loaded successfully ...')


class MAPPOMemory:
    """Buffer to store experience tuples."""
    def __init__(self, state_size: int, action_size: int, num_agents: int, single_state_space: bool, batch_size: int):
        """
        Initialize a SequentialMemory object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in multi-agent implementation
            single_state_space (bool): whether state space for each agent is single (17) or linear (17*n)
            batch_size (int): size of each training batch
        """
        self.memory = []
        self.experience = namedtuple("Experience", field_names=["states", "actions", "probs", "vals", "rewards", "dones"])

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.single_state_space = single_state_space
        self.batch_size = batch_size

    def store_memory(self, states, actions, probs, vals, rewards, dones):
        """Add new experiences to memory."""
        e = self.experience(states, actions, probs, vals, rewards, dones)
        self.memory.append(e)
    
    def clear_memory(self):
        self.memory = []

    def generate_batches(self):
        """Generate batches of experiences from memory."""
        n_states = len(self.memory)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        states = np.array([e.states for e in self.memory])
        actions = np.array([e.actions for e in self.memory])
        probs = np.array([e.probs for e in self.memory])
        vals = np.array([e.vals for e in self.memory])
        rewards = np.array([e.rewards for e in self.memory])
        dones = np.array([e.dones for e in self.memory])
        whole_states = states.reshape(-1, self.num_agents * self.state_size) if self.single_state_space else None
  
        return states, actions, probs, vals, rewards, dones, whole_states, batches

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
