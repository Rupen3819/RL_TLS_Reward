import numpy as np

from model import ActorNet, CriticNet

import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

#BATCH_SIZE = 256        # minibatch size
#N_EPOCHS = 10           # the optimizer’s number of epochs
#POLICY_CLIP = 0.1       # clipping parameter epsilon
#GAMMA = 0.99            # discount factor
#GAE_LAMBDA = 0.95       # factor for trade-off of bias vs variance for Generalized Advantage Estimator
#P_LR = 1e-4             # critic net learning rate
#C_LR = 1e-3             # policy net learning rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent:
    def __init__(self, state_size, action_size, actor_dim, critic_dim, fixed_action_space, TL_list, BATCH_SIZE, N_EPOCHS,
                 POLICY_CLIP, GAMMA, GAE_LAMBDA, P_LR, C_LR):
        self.state_size = state_size + len(TL_list) if fixed_action_space else state_size
        self.action_size = action_size
        self.fixed_action_space = fixed_action_space

        self.TL_list = TL_list
        self.BATCH_SIZE = BATCH_SIZE
        self.N_EPOCHS = N_EPOCHS
        self.POLICY_CLIP = POLICY_CLIP
        self.GAMMA = GAMMA
        self.GAE_LAMBDA = GAE_LAMBDA
        self.P_LR = P_LR
        self.C_LR = C_LR

        print(actor_dim)
        actor_dim.insert(0, self.state_size)
        actor_dim.append(self.action_size)
        self.actor = ActorNet(actor_dim, 'ppo').to(device)
        print(self.actor)

        print(critic_dim)
        critic_dim.insert(0, self.state_size)
        critic_dim.append(1)
        self.critic = CriticNet(critic_dim, 'ppo').to(device)
        print(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=P_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=C_LR)

        self.memory = PPOMemory(self.BATCH_SIZE)

    def remember(self, state, action, probs, vals, reward, done):
        reward = sum(list(reward.values()))
        if self.fixed_action_space:
            for tl_index, tl_id in enumerate(self.TL_list):
                one_hot = np.zeros(len(self.TL_list))
                np.put(one_hot, tl_index, 1)
                state_one_hot = list(np.array(one_hot.tolist() + state))

                self.memory.store_memory(state_one_hot, action[tl_id], probs[tl_id], vals[tl_id], reward, done)
        else:
            self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        if self.fixed_action_space:
            action_dict = dict.fromkeys(self.TL_list, 0)
            probs_dict = dict.fromkeys(self.TL_list, 0)
            value_dict = dict.fromkeys(self.TL_list, 0)

            for tl_index, tl_id in enumerate(self.TL_list):
                one_hot = np.zeros(len(self.TL_list))
                np.put(one_hot, tl_index, 1)
                state_one_hot = list(np.array(one_hot.tolist() + observation))
                state_one_hot = T.tensor([state_one_hot], dtype=T.float).to(device)

                dist = self.actor(state_one_hot)
                value = self.critic(state_one_hot)
                action = dist.sample()

                probs_dict[tl_id] = T.squeeze(dist.log_prob(action)).item()
                action_dict[tl_id] = T.squeeze(action).item()
                value_dict[tl_id] = T.squeeze(value).item()

            return action_dict, probs_dict, value_dict
        else:
            state = T.tensor([observation], dtype=T.float).to(device)

            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()

            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
            value = T.squeeze(value).item()

            return action, probs, value

    def learn(self):
        for _ in range(self.N_EPOCHS):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.GAMMA * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.GAMMA * self.GAE_LAMBDA
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(device)

            values = T.tensor(values).to(device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(device)
                old_probs = T.tensor(old_prob_arr[batch]).to(device)
                actions = T.tensor(action_arr[batch]).to(device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.POLICY_CLIP,
                                                 1 + self.POLICY_CLIP) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
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

    def load_models(self, path):
        print('... loading models ...')
        self.actor.load_checkpoint(path)
        self.critic.load_checkpoint(path)



class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
