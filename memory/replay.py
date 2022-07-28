import math
import random
from collections import namedtuple

import numpy as np
import torch

from memory.structures import MinTree, SumTree, RingBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
MultiExperience = namedtuple('MultiExperience', field_names=['states', 'actions', 'reward', 'next_states', 'done'])
PpoExperience = namedtuple('PpoExperience', field_names=['state', 'action', 'reward', 'done', 'value', 'log_prob'])


class AbstractReplayBuffer:
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def add(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class ReplayBuffer(AbstractReplayBuffer):
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a SequentialMemory object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        super().__init__()
        self.memory = RingBuffer(buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory.data, k=self.batch_size)

        state_list, action_list, reward_list, next_state_list, done_list = list(zip(*experiences))

        states = torch.from_numpy(np.vstack(state_list)).float().to(device)
        actions = torch.from_numpy(np.vstack(action_list)).long().to(device)
        rewards = torch.from_numpy(np.vstack(reward_list)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_state_list)).float().to(device)
        dones = torch.from_numpy(np.vstack(done_list).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer:
    def __init__(self, max_len: int, batch_size: int, alpha: float, beta: float, priority_eps: float):
        self.max_len = max_len
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.priority_eps = priority_eps

        self.priority_sums = SumTree(self.max_len)
        self.priority_mins = MinTree(self.max_len)

        self.experiences = RingBuffer(max_len)

        self.max_priority = 1
        self.last_batch_indices = None

    def __len__(self):
        return len(self.experiences)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory, along with its priority"""
        # Get the index of the experience (once it's added)
        index = (self.experiences.start + len(self)) % self.max_len

        e = Experience(state, action, reward, next_state, done)
        self.experiences.append(e)

        reduced_max_priority = self.max_priority ** self.alpha

        # Update the min/sum segment trees
        self.priority_sums.update(index, reduced_max_priority)
        self.priority_mins.update(index, reduced_max_priority)

    def sample(self):
        priority_sum = self.priority_sums.get_root()
        priority_min = self.priority_mins.get_root()

        # Compute the max weight, which occurs for the experience least likely to be sampled (has min priority)
        min_prob = priority_min / priority_sum
        max_weight = (len(self) * min_prob) ** -self.beta

        samples = [None] * self.batch_size

        for i in range(self.batch_size):
            # Get a priority in the interval [0, priority_sum)
            p = random.random() * priority_sum

            # Get the index of the experience whose ?
            index = self.priority_sums.get_prefix_sum_index(p)

            # Compute the probability of the experience
            prob = self.priority_sums.get_value(index) / priority_sum

            # Calculate the importance-sampling weight, normalizing by the max weight
            weight = (len(self) * prob) ** -self.beta / max_weight

            samples[i] = (*self.experiences[index], index, weight)

        state_list, action_list, reward_list, next_state_list, done_list, index_list, weight_list = list(zip(*samples))

        states = torch.from_numpy(np.vstack(state_list)).float().to(device)
        actions = torch.from_numpy(np.vstack(action_list)).long().to(device)
        rewards = torch.from_numpy(np.vstack(reward_list)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_state_list)).float().to(device)
        dones = torch.from_numpy(np.vstack(done_list).astype(np.uint8)).float().to(device)
        indices = torch.from_numpy(np.vstack(index_list)).to(device)
        weights = torch.from_numpy(np.vstack(weight_list)).float().to(device)

        self.last_batch_indices = indices

        return states, actions, rewards, next_states, dones, weights

    def update_priorities(self, new_priorities):
        """Replaces the priorities of the last sampled batch with new priorities"""
        if self.last_batch_indices is None:
            raise RuntimeError()

        new_priorities = new_priorities.abs() + self.priority_eps
        self.max_priority = max(self.max_priority, new_priorities.max().item())
        reduced_priorities = new_priorities ** self.alpha

        for i in range(self.batch_size):
            index, reduced_priority = self.last_batch_indices[i].item(), reduced_priorities[i].item()
            self.priority_sums.update(index, reduced_priority)
            self.priority_mins.update(index, reduced_priority)

        self.last_batch_indices = None


class NStepReplayBuffer(AbstractReplayBuffer):
    def __init__(self, replay_buffer, n_step, gamma, step_buffers=1):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.n_step = n_step
        self.gamma = gamma
        self.past_experiences = [RingBuffer(max_len=n_step) for _ in range(step_buffers)]
        self.step_buffers = step_buffers
        self.buffer_index = 0

    def __len__(self):
        return len(self.replay_buffer)

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        past_experiences = self.past_experiences[self.buffer_index]

        # Add the latest experience to the buffer, discarding the oldest experience if full
        # (which will already have been processed)
        past_experiences.append(experience)

        if done:
            # Process all remaining experiences in the buffer
            for i in range(len(past_experiences)):
                self._process_experience(past_experiences, i, next_state)
        elif len(past_experiences) >= self.n_step:
            # Process the oldest experience in the buffer
            self._process_experience(past_experiences, 0, next_state)

        self.buffer_index = (self.buffer_index + 1) % self.step_buffers

    def sample(self):
        return self.replay_buffer.sample()

    def _process_experience(self, past_experiences, experience_index, new_next_state):
        total_reward = 0

        # Compute total reward
        for i in range(len(past_experiences) - experience_index):
            past_experience = past_experiences[i + experience_index]
            reward = past_experience.reward
            total_reward += reward * math.pow(self.gamma, i)

        # Store oldest experience in replay buffer, with its new N-step reward and new next state
        (past_state, past_action, _, _, past_done) = past_experiences[experience_index]
        self.replay_buffer.add(past_state, past_action, total_reward, new_next_state, past_done)


class MultiSequentialMemory:
    """Fixed-size buffer to store multi-agent experience tuples."""

    def __init__(self, buffer_size, batch_size, single_state_space, local_reward_signal):
        """Initialize a SequentialMemory object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            single_state_space (bool): whether state space for each agent is single (17) or linear (17*n)
            local_reward_signal (bool): whether reward signal for each agent is local (r) or global (sum(r))

        """
        self.memory = RingBuffer(buffer_size)
        self.batch_size = batch_size

        self.states_func = np.array if single_state_space else np.vstack
        self.rewards_func = np.array if local_reward_signal else np.vstack

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, states, actions, reward, next_states, done):
        """Add a new experience to memory."""
        e = MultiExperience(states, actions, reward, next_states, done)

        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory.data, k=self.batch_size)

        states_list, actions_list, rewards_list, next_states_list, dones_list = list(zip(*experiences))

        states = torch.from_numpy(self.states_func(states_list)).float().to(device)
        next_states = torch.from_numpy(self.states_func(next_states_list)).float().to(device)
        actions = torch.from_numpy(np.array(actions_list)).long().to(device)
        rewards = torch.from_numpy(self.rewards_func(rewards_list)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones_list).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones


class BatchMemory:
    def __init__(self, batch_size):
        self.experiences = []
        self.batch_size = batch_size

    def store(self, state, action, reward, done, values, probs):
        experience = PpoExperience(state, action, reward, done, values, probs)
        self.experiences.append(experience)

    def reset(self):
        self.experiences = []

    def generate_batches(self):
        indices = np.arange(len(self.experiences), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(self.experiences), self.batch_size)
        ]

        return *map(np.array, zip(*self.experiences)), batches
