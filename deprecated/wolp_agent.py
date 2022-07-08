import numpy as np

from deprecated.ddpg_agent import DDPGAgent
import deprecated.action_space as action_space

import torch
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WolpertingerAgent(DDPGAgent):
    def __init__(self, state_size, action_size, actor_dim, critic_dim, epsilon, actor_init_w, critic_init_w, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, OU_THETA, OU_MU, OU_SIGMA, P_LR, V_LR, WEIGHT_DECAY, k_ratio=0.1):
        super().__init__(state_size, action_size, actor_dim, critic_dim, epsilon, actor_init_w, critic_init_w, BUFFER_SIZE, BATCH_SIZE, OU_THETA, OU_MU, OU_SIGMA, P_LR, V_LR, WEIGHT_DECAY)

        self.GAMMA = GAMMA
        self.TAU = TAU

        self.action_space = action_space.Discrete_space(action_size)
        self.k_nearest_neighbors = max(1, int(action_size * k_ratio)) if action_size < int(1e6) else max(1, int(action_size * pow(k_ratio, 2)))

    def get_action_space(self):
        return self.action_space

    def normalize_proto_action(self, action):
        return (action + 1 ) / 2

    def reverse_normalized_action(self, action):
        return action * 2 - 1

    def select_action(self, state, decay_epsilon=True):
        # Taking a continuous action from the actor
        proto_action = super().select_action(state, decay_epsilon)
        proto_action = self.normalize_proto_action(proto_action)

        raw_wolp_action, wolp_action = self.wolp_action(state, proto_action)
        raw_wolp_action = raw_wolp_action[0]

        # Return the best neighbor of the proto action, this is an action for env step
        return raw_wolp_action, wolp_action[0]

    def random_action(self):
        # Taking a random continuous action
        proto_action = super().random_action()

        raw_action, action = self.action_space.search_point(proto_action, 1)
        raw_action = raw_action[0]
        raw_action = self.reverse_normalized_action(raw_action)

        action = action[0]

        return raw_action, action[0]

    def select_target_action(self, state):
        self.actor_target.eval()
        with torch.no_grad():
            proto_action = self.actor_target(state)
        self.actor_target.train()

        proto_action = torch.clamp(proto_action, -1.0, 1.0).cpu().data.numpy().astype(np.float64)
        proto_action = self.normalize_proto_action(proto_action)

        raw_wolp_action, wolp_action = self.wolp_action(state, proto_action, True)

        return raw_wolp_action

    def wolp_action(self, state, proto_action, is_target_update=False):
        # Get the proto_action's k nearest neighbors
        raw_actions, actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)

        if not isinstance(state, np.ndarray):
            state = state.cpu().data.numpy()

        # Make all the state, action pairs for the critic
        if is_target_update:
            def tile_along_axis(state):
                return np.tile(state, [raw_actions.shape[1], 1])

            states = np.apply_along_axis(tile_along_axis, axis=1, arr=state)
        else:
            states = np.tile(state, [raw_actions.shape[1], 1])
            states = states.reshape(len(raw_actions), raw_actions.shape[1], states.shape[1]) if self.k_nearest_neighbors > 1 else states.reshape(raw_actions.shape[0], states.shape[1])

        raw_actions = self.reverse_normalized_action(raw_actions)
        raw_actions = torch.from_numpy(raw_actions).float().to(device)
        states = torch.from_numpy(states).float().to(device)

        # Evaluate each pair through the critic
        self.critic_local.eval()
        with torch.no_grad():
            actions_evaluation = self.critic_local([states, raw_actions])
        self.critic_local.train()

        # Find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation.cpu().data.numpy().astype(np.float64), axis=1)
        max_index = max_index.reshape(len(max_index),)

        raw_actions = raw_actions.cpu().data.numpy().astype(np.float64)

        # Return the best action, i.e., wolpertinger action from the full wolpertinger policy
        if self.k_nearest_neighbors > 1:
            return raw_actions[[i for i in range(len(raw_actions))], max_index, [0]].reshape(len(raw_actions),1), actions[[i for i in range(len(actions))], max_index, [0]].reshape(len(actions),1)
        else:
            return raw_actions[max_index], actions[max_index]

    def update_policy(self):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1).to(device)

        self.critic_optimizer.zero_grad()
        next_wolp_action = self.select_target_action(next_states)
        next_wolp_action = torch.from_numpy(next_wolp_action).float().to(device)

        # Get max predicted Q values (for next states) from target critic model
        Q_targets_next = self.critic_target([next_states, next_wolp_action]).detach()

        # Compute Q targets for current states
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Q_targets = torch.clamp(Q_targets, -10.0, 10.0)

        # Get expected Q values from local critic model
        Q_expected = self.critic_local([states, actions])

        # Compute value loss
        value_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the value loss
        value_loss.backward()
        self.critic_optimizer.step()

        # Compute policy loss
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic_local([states, self.actor_local(states)])
        policy_loss = policy_loss.mean()

        # Minimize the policy loss
        policy_loss.backward()

        # for param in self.actor_local.parameters():
            # param.grad.data.clamp_(-1, 1)
        # self.actor_local.clip_gradients(10.0)
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local, self.actor_target, self.TAU)
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
