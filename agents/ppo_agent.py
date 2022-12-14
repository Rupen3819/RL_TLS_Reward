import numpy as np
import torch
import torch.optim as optim

from memory.replay import BatchMemory
from model import PpoActorNet, PpoCriticNet, PpoContinuousActorNet, PpoContinuousCriticNet
from settings import ActionDefinition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(
            self,
            state_size: int,
            action_size: int,
            actor_dim: list[int, ...],
            critic_dim: list[int, ...],
            fixed_action_space: bool = False,
            traffic_lights: dict[str, str] = None,
            batch_size: int = 256,
            n_epochs: int = 10,
            policy_clip: float = 0.1,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            policy_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-3,
            learning_interval: int = 64,
            action_definition: ActionDefinition = ActionDefinition.PHASE
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
            learning_interval (int): how often to learn from experiences
            action_definition (enum):
            green_time (int):
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
        self.learning_interval = learning_interval
        self.action_definition = action_definition

        if action_definition == ActionDefinition.PHASE:
            actor_class = PpoActorNet
            critic_class = PpoCriticNet
        else:  # action_definition == ActionDefinition.CYCLE
            actor_class = PpoContinuousActorNet
            critic_class = PpoContinuousCriticNet

        actor_structure = [self.state_size, *actor_dim, self.action_size]
        self.actor = actor_class('ppo', actor_structure).to(device)
        print(self.actor)

        critic_structure = [self.state_size, *critic_dim, 1]
        self.critic = critic_class('ppo', critic_structure).to(device)
        print(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = BatchMemory(self.batch_size)

        self.t_step = 0

    def _one_hot_state(self, index, state):
        one_hot = np.zeros(len(self.traffic_lights))
        np.put(one_hot, index, 1)
        state_one_hot = list(np.array(one_hot.tolist() + state))
        return state_one_hot

    def step(self, state, action, reward, done, value, log_prob):
        reward = sum(reward.values())

        # Save experience in batch memory
        if self.fixed_action_space:
            for index, light_id in enumerate(self.traffic_lights):
                state_one_hot = self._one_hot_state(index, state)
                self.memory.store(state_one_hot, action[light_id], reward, done, value[light_id], log_prob[light_id])
        else:
            self.memory.store(state, action, reward, done, value, log_prob)

        # Learn every update_interval time steps.
        self.t_step += 1
        if self.t_step % self.learning_interval == 0:
            self._learn()

    def act(self, observation):
        if self.fixed_action_space:
            actions = {}
            values = {}
            log_probs = {}

            for index, light_id in enumerate(self.traffic_lights):
                state_one_hot = torch.tensor([self._one_hot_state(index, observation)], dtype=torch.float).to(device)
                actions[light_id], values[light_id], log_probs[light_id] = self._choose_action(state_one_hot)

            return actions, values, log_probs

        else:
            state = torch.tensor([observation], dtype=torch.float).to(device)
            return self._choose_action(state)

    def _choose_action(self, state):
        dist = self.actor(state)
        value = self.critic(state)

        value = torch.squeeze(value).item()

        if self.action_definition == ActionDefinition.PHASE:
            action = dist.sample()
            log_prob = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            return action, value, log_prob
        else:  # self.action_definition == ActionDefinition.CYCLE
            raw_action = dist.sample().squeeze()
            log_prob = dist.log_prob(raw_action).sum(axis=1)
            log_prob = log_prob.detach().item()
            raw_action = raw_action.tolist()
            return raw_action, value, log_prob

    def _learn(self):
        for _ in range(self.n_epochs):
            states, actions, rewards, dones, values, log_probs, batches = self.memory.generate_batches()

            advantages = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                advantage = 0

                for k in range(t, len(rewards) - 1):
                    advantage += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda

                advantages[t] = advantage

            advantages = torch.tensor(advantages).to(device)
            values = torch.tensor(values).to(device)

            for batch in batches:
                states_batch = torch.tensor(states[batch], dtype=torch.float).to(device)

                actions_dtype = torch.int if self.action_definition == ActionDefinition.PHASE else torch.float
                actions_batch = actions[batch]
                actions_batch = torch.tensor(actions_batch, dtype=actions_dtype).to(device)

                log_probs_batch = torch.tensor(log_probs[batch], dtype=torch.float).to(device)

                dist = self.actor(states_batch)
                critic_value = self.critic(states_batch)
                critic_value = torch.squeeze(critic_value)

                if self.action_definition == ActionDefinition.PHASE:
                    new_log_probs_batch = dist.log_prob(actions_batch)
                else:  # self.action_definition == ActionDefinition.CYCLE
                    new_log_probs_batch = dist.log_prob(actions_batch).sum(axis=1)

                prob_ratio = (new_log_probs_batch - log_probs_batch).exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * advantages[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantages[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.reset()

    def save_model(self, path):
        print('Saving model')
        self.actor.save_checkpoint(path)
        self.critic.save_checkpoint(path)
        print('Model saved successfully')

    def load_model(self, path):
        print('Loading model')
        self.actor.load_checkpoint(path)
        self.critic.load_checkpoint(path)
        print('Model loaded successfully')
