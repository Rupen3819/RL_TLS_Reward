import numpy as np
import torch
import torch.optim as optim

from agents.replay import PPOMemory
from model import PpoActorNet, PpoCriticNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(
            self, state_size: int, action_size: int, actor_dim: tuple[int, ...], critic_dim: tuple[int, ...],
            fixed_action_space: bool, traffic_lights: dict[str, str], batch_size: int = 256, n_epochs: int = 10,
            policy_clip: float = 0.1, gamma: float = 0.99, gae_lambda: float = 0.95, policy_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3
    ):
        """
        Initialize MADQN Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            actor_dim: (tuple): dimensions of hidden layers for actor network
            critic_dim: (tuple): dimensions of hidden layers for critic network
            fixed_action_space (bool): whether action space is linear (8) or exponential (8^n)
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
        if fixed_action_space:
            self.state_size += len(traffic_lights)

        self.action_size = action_size
        self.fixed_action_space = fixed_action_space

        self.traffic_lights = traffic_lights

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate

        print(actor_dim)
        actor_dim = (self.state_size, actor_dim[0], actor_dim[1], self.action_size)
        self.actor = PpoActorNet('ppo', actor_dim).to(device)
        print(self.actor)

        print(critic_dim)
        critic_dim = (self.state_size, critic_dim[0], critic_dim[1], 1)
        self.critic = PpoCriticNet('ppo', critic_dim).to(device)
        print(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = PPOMemory(self.batch_size)

    def remember(self, state, action, probs, values, reward, done):
        reward = sum(reward.values())

        if self.fixed_action_space:
            for index, traffic_light_id in enumerate(self.traffic_lights):
                one_hot = np.zeros(len(self.traffic_lights))
                np.put(one_hot, index, 1)
                state_one_hot = list(np.array(one_hot.tolist() + state))

                self.memory.store_memory(
                    state_one_hot, action[traffic_light_id], probs[traffic_light_id], values[traffic_light_id], reward, done
                )
        else:
            self.memory.store_memory(state, action, probs, values, reward, done)

    def choose_action(self, observation):
        if self.fixed_action_space:
            actions = dict.fromkeys(self.traffic_lights, 0)
            probs = dict.fromkeys(self.traffic_lights, 0)
            values = dict.fromkeys(self.traffic_lights, 0)

            for index, traffic_light_id in enumerate(self.traffic_lights):
                one_hot = np.zeros(len(self.traffic_lights))
                np.put(one_hot, index, 1)
                state_one_hot = list(np.array(one_hot.tolist() + observation))
                state_one_hot = torch.tensor([state_one_hot], dtype=torch.float).to(device)

                dist = self.actor(state_one_hot)
                value = self.critic(state_one_hot)
                action = dist.sample()

                probs[traffic_light_id] = torch.squeeze(dist.log_prob(action)).item()
                actions[traffic_light_id] = torch.squeeze(action).item()
                values[traffic_light_id] = torch.squeeze(value).item()

            return actions, probs, values

        else:
            state = torch.tensor([observation], dtype=torch.float).to(device)

            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()

            probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()

            return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, rewards, dones, batches = self.memory.generate_batches()

            values = vals_arr
            advantages = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = a_t

            advantages = torch.tensor(advantages).to(device)
            values = torch.tensor(values).to(device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(device)
                actions = torch.tensor(action_arr[batch]).to(device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * advantages[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantages[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

    def save_models(self, path):
        print('... saving models ...')
        self.actor.save_checkpoint(path)
        self.critic.save_checkpoint(path)
        print('... models saved successfully ...')

    def load_models(self, path):
        print('... loading models ...')
        self.actor.load_checkpoint(path)
        self.critic.load_checkpoint(path)
        print('... models loaded successfully')
