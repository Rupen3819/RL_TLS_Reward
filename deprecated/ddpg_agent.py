import numpy as np
import torch
import torch.optim as optim

from memory.replay import ReplayBuffer
from deprecated.random_process import OrnsteinUhlenbeckProcess
from model import WolpActorNet, WolpCriticNet

#BUFFER_SIZE = int(1e5)  # replay buffer size
#BATCH_SIZE = 64         # minibatch size
#GAMMA = 0.99            # discount factor
#OU_THETA = 0.15         # noise theta
#OU_MU = 0.0             # noise mu
#OU_SIGMA = 0.2          # noise sigma
#P_LR = 1e-4             # critic net learning rate
#C_LR = 1e-3             # policy net learning rate
#WEIGHT_DECAY = 1e-4     # weight decay for L2 Regularization loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, actor_dim, critic_dim, epsilon, actor_init_w, critic_init_w, BUFFER_SIZE, BATCH_SIZE, OU_THETA, OU_MU, OU_SIGMA, P_LR, V_LR, WEIGHT_DECAY):
        """Initialize a DDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            epsilon (int): linear decay of exploration policy
            actor_init_w (float): for weight initialization of last actor layer
            critic_init_w (float): for weight initialization of last critic layer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_dim = 1
        self.depsilon = 1.0 / epsilon
        self.epsilon = 1.0
        self.actor_init_w = actor_init_w
        self.critic_init_w = critic_init_w

        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.OU_THETA = OU_THETA
        self.OU_MU = OU_MU
        self.OU_SIGMA = OU_SIGMA
        self.P_LR = P_LR
        self.V_LR = V_LR
        self.WEIGHT_DECAY = WEIGHT_DECAY


        # Actor-Critic-Network
        print(actor_dim)
        print(critic_dim)
        actor_dim.insert(0, self.state_size)
        actor_dim.append(self.action_dim)
        critic_dim.insert(0, self.state_size)
        critic_dim.append(1)

        self.actor_local = WolpActorNet(actor_dim, self.actor_init_w).to(device)
        print(self.actor_local)
        self.actor_target = WolpActorNet(actor_dim, self.actor_init_w).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=P_LR, weight_decay=WEIGHT_DECAY)
        # self.actor_optimizer = optim.RAdam(self.actor_local.parameters(), lr=P_LR, weight_decay=WEIGHT_DECAY)

        self.critic_local = WolpCriticNet(critic_dim, 'wolp', self.critic_init_w).to(device)
        print(self.critic_local)
        self.critic_target = WolpCriticNet(critic_dim, 'wolp', self.critic_init_w).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=V_LR, weight_decay=WEIGHT_DECAY)
        # self.critic_optimizer = optim.RAdam(self.critic_local.parameters(), lr=V_LR, weight_decay=WEIGHT_DECAY)

        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Replay memory
        self.memory = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_dim, theta=OU_THETA, mu=OU_MU, sigma=OU_SIGMA)

        # Initialize time step (for start learning policy and updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self):
        self.t_step += 1
    
    def observe(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        reward = sum(list(reward.values()))
        self.memory.add(state, action, reward, next_state, done)

    def random_action(self):
        # Taking a random continuous proto action for wolp action
        # action = np.random.uniform(-1., 1., self.action_dim)
        action = np.random.uniform(0.0, 1.0, self.action_dim)

        return action

    def select_action(self, state, decay_epsilon=True):
        # Taking a continuous proto action from the actor for wolp action
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()

        action = action.cpu().data.numpy().astype(np.float64)
        action += max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1.0, 1.0)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        return action
    
    def reset(self):
        self.random_process.reset_states()

    def update_policy(self):
        pass

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_model(self, path):
        print('... saving models ...')
        self.actor_local.save_checkpoint(path)
        self.critic_local.save_checkpoint(path)

    def load_model(self, path):
        print('... loading models ...')
        self.actor_local.load_checkpoint(path)
        self.critic_local.load_checkpoint(path)