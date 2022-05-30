import numpy as np

from collections import namedtuple

from model import ActorNet, CriticNet

import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim

#BATCH_SIZE = 256        # minibatch size
#N_EPOCHS = 10           # the optimizer’s number of epochs
#POLICY_CLIP = 0.1       # clipping parameter epsilon
#GAMMA = 0.99            # discount factor
#GAE_LAMBDA = 0.95       # factor for trade-off of bias vs variance for Generalized Advantage Estimator
#P_LR = 1e-4             # critic net learning rate
#C_LR = 1e-3             # policy net learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAPPOAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, actor_dim, critic_dim, training_strategy, actor_parameter_sharing, critic_parameter_sharing,
                 single_state_space, local_reward_signal, TL_list, BATCH_SIZE, N_EPOCHS, POLICY_CLIP, GAMMA, GAE_LAMBDA, P_LR, C_LR):
        """Initialize MAPPO Agent objects.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in multi-agent implementation
            single_state_space (bool): whether state space for each agent is single (17) or linear (17*n)
            local_reward_signal (bool): whether reward signal for each agent is local (r) or global (sum(r))
            TL_list (dict): traffic light ids from the SUMO environment
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
        """
        self.state_size = int(state_size / num_agents) if single_state_space else state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.training_strategy = training_strategy
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing
        self.single_state_space = single_state_space
        self.local_reward_signal = local_reward_signal

        self.TL_list = TL_list
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.POLICY_CLIP = POLICY_CLIP
        self.GAMMA = GAMMA
        self.GAE_LAMBDA = GAE_LAMBDA
        self.P_LR = P_LR
        self.C_LR = C_LR


        # Actor-Critic-Networks
        print(actor_dim)
        print(critic_dim)
        actor_dim.insert(0, self.state_size)
        actor_dim.append(self.action_size)
        critic_dim.insert(0, self.state_size)
        critic_dim.append(1)

        self.actors = [ActorNet(actor_dim, 'mappo').to(device)] * self.num_agents
        print(self.actors)
        if self.training_strategy == 'nonstrategic':
            self.critics = [CriticNet(critic_dim, 'mappo').to(device)] * self.num_agents
        elif self.training_strategy == 'concurrent':
            self.critics = [CriticNet(critic_dim, 'mappo').to(device)] * self.num_agents
        elif self.training_strategy == 'centralized':
            critic_state_dim = self.state_size * self.num_agents
            critic_dim[0] = critic_state_dim
            self.critics = [CriticNet(critic_dim, 'mappo').to(device)] * self.num_agents
        print(self.critics)

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=P_LR) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=C_LR) for c in self.critics]

        # Tricky and memory consuming implementation of parameter sharing
        if not self.training_strategy == 'nonstrategic':
            if self.actor_parameter_sharing:
                for agent_id in range(1, self.num_agents):
                    self.actors[agent_id] = self.actors[0]
                    self.actor_optimizers[agent_id] = self.actor_optimizers[0]
            if self.critic_parameter_sharing:
                for agent_id in range(1, self.num_agents):
                    self.critics[agent_id] = self.critics[0]
                    self.critic_optimizers[agent_id] = self.critic_optimizers[0]

        # Replay memory
        self.memory = SequentialMemory(self.state_size, self.action_size, self.num_agents, self.single_state_space, self.BATCH_SIZE)

    def remember(self, states, actions, probs, vals, rewards, dones):
        # Save experience in replay memory
        if self.single_state_space:
            states = [states[i:i + self.state_size] for i in range(0, len(states), self.state_size)]
        else:
            states = states
        if self.local_reward_signal:
            rewards = list(rewards.values())
        else:
            rewards = sum(list(rewards.values()))
        actions = list(actions.values())

        self.memory.store_memory(states, actions, probs, vals, rewards, dones)

    def choose_action(self, observation):
        # Returns actions, probs and values for given observation as per current policy
        if self.single_state_space:
            state = T.tensor([observation], dtype=T.float).view(self.num_agents, -1).to(device)
            whole_state = state.view(-1, self.state_size * self.num_agents)
        else:
            state = T.tensor([observation], dtype=T.float).to(device)

        actions = dict.fromkeys(self.TL_list, 0)
        probs = [0] * self.num_agents
        values = [0] * self.num_agents

        for agent_id in range(self.num_agents):
            if self.single_state_space:
                dist = self.actors[agent_id](state[agent_id, :])
                if self.training_strategy == 'nonstrategic':
                    value = self.critics[agent_id](state[agent_id, :])
                elif self.training_strategy == 'concurrent':
                    value = self.critics[agent_id](state[agent_id, :])
                elif self.training_strategy == 'centralized':
                    value = self.critics[agent_id](whole_state)
            else:
                dist = self.actors[agent_id](state)
                value = self.critics[agent_id](state)
            action = dist.sample()

            probs[agent_id] = T.squeeze(dist.log_prob(action)).item()
            actions[f'TL{agent_id + 1}'] = T.squeeze(action).item()
            values[agent_id] = T.squeeze(value).item()

        return actions, probs, values

    def learn(self):
        for _ in range(self.N_EPOCHS):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, \
                whole_state_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros((len(reward_arr), self.num_agents), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                action = 0
                for k in range(t, len(reward_arr) - 1):
                    action += discount * (reward_arr[k] + self.GAMMA * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.GAMMA * self.GAE_LAMBDA
                for agent_id in range(self.num_agents):
                    advantage[t][agent_id] = action[agent_id]
            advantage = T.tensor(advantage).to(device)

            values = T.tensor(values).to(device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(device)
                if self.single_state_space:
                    whole_states = T.tensor(whole_state_arr[batch], dtype=T.float).to(device)
                old_probs = T.tensor(old_prob_arr[batch]).to(device)
                actions = T.tensor(action_arr[batch]).to(device)

                for agent_id in range(self.num_agents):
                    if self.single_state_space:
                        dist = self.actors[agent_id](states[:, agent_id])
                        if self.training_strategy == 'nonstrategic':
                            critic_value = self.critics[agent_id](states[:, agent_id])
                        elif self.training_strategy == 'concurrent':
                            critic_value = self.critics[agent_id](states[:, agent_id])
                        elif self.training_strategy == 'centralized':
                            critic_value = self.critics[agent_id](whole_states)
                    else:
                        dist = self.actors[agent_id](states)
                        critic_value = self.critics[agent_id](states)
                    critic_value = T.squeeze(critic_value)

                    new_probs = dist.log_prob(actions[:, agent_id])
                    prob_ratio = new_probs.exp() / old_probs[:, agent_id].exp()
                    weighted_probs = advantage[batch][:, agent_id] * prob_ratio
                    weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.POLICY_CLIP, 1 + self.POLICY_CLIP) * advantage[batch][:, agent_id]
                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

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

    def load_models(self, path):
        print('... loading models ...')
        for agent_id in range(self.num_agents):
            self.actors[agent_id].load_checkpoint(path, f'{agent_id}')
            self.critics[agent_id].load_checkpoint(path, f'{agent_id}')



class SequentialMemory:
    """Buffer to store experience tuples."""
    def __init__(self, state_size, action_size, num_agents, single_state_space, batch_size):
        """Initialize a SequentialMemory object.

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
