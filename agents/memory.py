import math
import random
from collections import namedtuple

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RingBuffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.start = 0
        self.length = 0
        self.data = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError('Buffer index out of range')
        return self.data[(self.start + index) % self.max_len]

    def __setitem__(self, index, value):
        if index < 0 or index >= self.length:
            raise IndexError('Buffer index out of range')
        self.data[(self.start + index) % self.max_len] = value

    def __iter__(self):
        if self.length == 0:
            return

        index = self.start
        end_index = self.start

        finished = False
        while not finished:
            yield self.data[index]
            index = (index + 1) % self.length
            finished = index == end_index

    def append(self, value):
        if self.length < self.max_len:
            self.length += 1
            self.data.append(value)
        elif self.length == self.max_len:
            self.start = (self.start + 1) % self.max_len
            self.data[(self.start + self.length - 1) % self.max_len] = value


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a SequentialMemory object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = RingBuffer(buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.Experience(state, action, reward, next_state, done)
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


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, batch_size, n_step, gamma):
        super().__init__(buffer_size, batch_size)
        self.n_step = n_step
        self.gamma = gamma
        self.past_experiences = RingBuffer(max_len=n_step)

    def add(self, state, action, reward, next_state, done):
        experience = super().Experience(state, action, reward, next_state, done)

        # Add the latest experience to the buffer, discarding the oldest experience if full
        # (which will already have been processed)
        self.past_experiences.append(experience)

        if done:
            # Process all remaining experiences in the buffer
            for i in range(len(self.past_experiences)):
                self._process_experience(i, next_state)
        elif len(self.past_experiences) >= self.n_step:
            # Process the oldest experience in the buffer
            self._process_experience(0, next_state)

    def _process_experience(self, experience_index, new_next_state):
        total_reward = 0

        # Compute total reward
        for i in range(len(self.past_experiences) - experience_index):
            past_experience = self.past_experiences[i + experience_index]
            reward = past_experience.reward
            total_reward += reward * math.pow(self.gamma, i)

        # Store oldest experience in replay buffer, with its new N-step reward and new next state
        (past_state, past_action, _, _, past_done) = self.past_experiences[experience_index]
        super().add(past_state, past_action, total_reward, new_next_state, past_done)


class MultiSequentialMemory:
    """Fixed-size buffer to store multi-agent experience tuples."""

    MultiExperience = namedtuple("MultiExperience", field_names=["states", "actions", "reward", "next_states", "done"])

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
        e = self.MultiExperience(states, actions, reward, next_states, done)

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
