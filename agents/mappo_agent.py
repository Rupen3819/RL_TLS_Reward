from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim

from environment import get_intersection_name
from memory.replay import BatchMultiMemory
from model import PpoActorNet, PpoCriticNet
from settings import TrainingStrategy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAPPOAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self,
            state_size: int,
            action_size: int,
            num_agents: int,
            actor_dim: list[int, ...],
            critic_dim: list[int, ...],
            training_strategy: TrainingStrategy,
            actor_parameter_sharing: bool,
            critic_parameter_sharing: bool,
            single_state_space: bool,
            local_reward_signal: bool,
            traffic_lights: dict[str, str],
            batch_size: int = 256,
            n_epochs: int = 10,
            policy_clip: int = 0.1,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            policy_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3,
            learning_interval: int = 64
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
            n_epochs (int): the optimizer???s number of epochs
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
        self.learning_interval = learning_interval

        # Actor-Critic-Networks
        actor_dim = (self.state_size, *actor_dim, self.action_size)

        if self.training_strategy == TrainingStrategy.CENTRALIZED:
            critic_state_dim = self.state_size * self.num_agents
            critic_dim = [critic_state_dim, *critic_dim, 1]
        else:
            critic_dim = [self.state_size, *critic_dim, 1]

        if self.actor_parameter_sharing and self.training_strategy != TrainingStrategy.NONSTRATEGIC:
            self.actors = [PpoActorNet('mappo', actor_dim).to(device) for _ in range(self.num_agents)]
            self.actor_optimizers = [optim.Adam(actor.parameters(), lr=policy_learning_rate) for actor in self.actors]
        else:
            self.actors = [PpoActorNet('mappo', actor_dim).to(device)] * self.num_agents
            self.actor_optimizers = [optim.Adam(self.actors[0].parameters(), lr=policy_learning_rate)] * self.num_agents
        print(self.actors)

        if self.critic_parameter_sharing and self.training_strategy != TrainingStrategy.NONSTRATEGIC:
            self.critics = [PpoCriticNet('mappo', critic_dim).to(device) for _ in range(self.num_agents)]
            self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_learning_rate) for critic in self.critics]
        else:
            self.critics = [PpoCriticNet('mappo', critic_dim).to(device)] * self.num_agents
            self.critic_optimizers = [optim.Adam(self.critics[0].parameters(), lr=critic_learning_rate)] * self.num_agents
        print(self.critics)

        # Replay memory
        self.memory = BatchMultiMemory(self.batch_size, self.single_state_space, self.state_size, self.action_size, self.num_agents)

        self.t_step = 0

    def step(self, states, actions, rewards, dones, values, probs):

        # Save experience in batch memory
        if self.single_state_space:
            states = [states[i:i + self.state_size] for i in range(0, len(states), self.state_size)]

        if self.local_reward_signal:
            rewards = list(rewards.values())
        else:
            rewards = sum(rewards.values())

        actions = list(actions.values())

        self.memory.store(states, actions, probs, values, rewards, dones)

        # Learn every update_interval time steps.
        self.t_step += 1
        if self.t_step % self.learning_interval == 0:
            self._learn()

    def act(self, observation):
        # Returns actions, probs and values for given observation as per current policy
        if self.single_state_space:
            state = torch.tensor([observation], dtype=torch.float).view(self.num_agents, -1).to(device)
            whole_state = state.view(-1, self.state_size * self.num_agents)
        else:
            state = torch.tensor([observation], dtype=torch.float).to(device)

        actions = {}
        values = [0] * self.num_agents
        probs = [0] * self.num_agents

        for agent_id in range(self.num_agents):
            if self.single_state_space:
                dist = self.actors[agent_id](state[agent_id, :])

                if self.training_strategy == TrainingStrategy.CENTRALIZED:
                    value = self.critics[agent_id](whole_state)
                else:
                    value = self.critics[agent_id](state[agent_id, :])

            else:
                dist = self.actors[agent_id](state)
                value = self.critics[agent_id](state)

            action = dist.sample()

            actions[get_intersection_name(agent_id)] = torch.squeeze(action).item()
            values[agent_id] = torch.squeeze(value).item()
            probs[agent_id] = torch.squeeze(dist.log_prob(action)).item()

        return actions, values, probs

    def _learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_probs, values, rewards, dones, whole_states, batches = self.memory.generate_batches()

            advantage = np.zeros((len(rewards), self.num_agents), dtype=np.float32)

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
                states_batch = torch.tensor(states[batch], dtype=torch.float).to(device)

                if self.single_state_space:
                    whole_states_batch = torch.tensor(whole_states[batch], dtype=torch.float).to(device)

                old_probs_batch = torch.tensor(old_probs[batch]).to(device)
                actions_batch = torch.tensor(actions[batch]).to(device)

                for agent_id in range(self.num_agents):
                    if self.single_state_space:
                        dist = self.actors[agent_id](states_batch[:, agent_id])

                        if self.training_strategy == TrainingStrategy.CENTRALIZED:
                            critic_value = self.critics[agent_id](whole_states_batch)
                        else:
                            critic_value = self.critics[agent_id](states_batch[:, agent_id])
                    else:
                        dist = self.actors[agent_id](states_batch)
                        critic_value = self.critics[agent_id](states_batch)

                    critic_value = torch.squeeze(critic_value)

                    new_probs = dist.log_prob(actions_batch[:, agent_id])
                    prob_ratio = (new_probs - old_probs_batch[:, agent_id]).exp()
                    weighted_probs = advantage[batch][:, agent_id] * prob_ratio
                    weighted_clipped_probs = torch.clamp(
                        prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                    ) * advantage[batch][:, agent_id]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch][:, agent_id] + values[batch][:, agent_id]
                    critic_loss = ((returns - critic_value) ** 2).mean()

                    total_loss = actor_loss + 0.5 * critic_loss

                    self.actor_optimizers[agent_id].zero_grad()
                    self.critic_optimizers[agent_id].zero_grad()
                    total_loss.backward()
                    self.actor_optimizers[agent_id].step()
                    self.critic_optimizers[agent_id].step()

        self.memory.reset()
    
    def save_model(self, path):
        print('Saving model')
        for agent_id in range(self.num_agents):
            self.actors[agent_id].save_checkpoint(path, f'{agent_id}')
            self.critics[agent_id].save_checkpoint(path, f'{agent_id}')
        print('Model saved successfully')

    def load_model(self, path):
        print('Loading model')
        for agent_id in range(self.num_agents):
            self.actors[agent_id].load_checkpoint(path, f'{agent_id}')
            self.critics[agent_id].load_checkpoint(path, f'{agent_id}')
        print('Model loaded successfully')


