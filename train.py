import traci
import timeit
import numpy as np
import random
from State import TL_list
from utils import *
import os
import sys
import datetime
import heapq
import pandas as pd
from pandas import DataFrame
from utils import import_train_configuration
from shutil import copyfile
from collections import deque
import torch
import matplotlib.pyplot as plt
from pathlib import Path


config = import_train_configuration(config_file='training_settings.ini')
if not config['is_train']:
    config = import_test_configuration(config_file_path=config['test_model_path_name'])
print(config)

if config['is_train']:
    from Environment.SUMO_train import SUMO
    env=SUMO()
else:
    from Environment.SUMO_test import SUMO
    from Statistics_vehicles import Statistic_Vehicles
    vehicle_stats = Statistic_Vehicles()
    env = SUMO(vehicle_stats)

print('State shape: ', env.num_states)
print('Number of actions: ', env.action_space.n)

if config['agent_type'] == 'DQN':
    from dqn_agent import DQNAgent
    agent = DQNAgent(env.num_states, env.action_space.n, config['hidden_dim'], config['fixed_action_space'], env.TL_list, config['memory_size_max'],
                config['batch_size'], config['gamma'], config['tau'], config['learning_rate'], config['target_update'],
                )

if config['agent_type'] == 'PPO':
    from ppo_agent import PPOAgent
    agent = PPOAgent(env.num_states, env.action_space.n, config['actor_dim'], config['critic_dim'], config['fixed_action_space'],
                env.TL_list, config['batch_size'], config['n_epochs'], config['policy_clip'], config['gamma'], config['gae_lambda'],
                config['policy_learning_rate'], config['value_learning_rate']
                )

if config['agent_type'] == 'WOLP':
    from wolp_agent import WolpertingerAgent
    agent = WolpertingerAgent(env.num_states, env.action_space.n, config['actor_dim'], config['critic_dim'], config['eps_policy'], config['actor_init_w'],
                config['critic_init_w'], config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'], config['ou_theta'], config['ou_mu'],
                config['ou_sigma'], config['policy_learning_rate'], config['value_learning_rate'], config['weight_decay']
                )

if config['agent_type'] == 'MADQN':
    from madqn_agent import MADQNAgent
    agent = MADQNAgent(env.num_states, env.action_space.n, len(env.TL_list), config['hidden_dim'], config['single_state_space'], config['local_reward_signal'],
                env.TL_list, config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'], config['learning_rate'], config['target_update']
                )

if config['agent_type'] == 'MAPPO':
    from mappo_agent import MAPPOAgent
    agent = MAPPOAgent(env.num_states, env.action_space.n, len(env.TL_list), config['actor_dim'], config['critic_dim'], config['training_strategy'], config['actor_parameter_sharing'],
                config['critic_parameter_sharing'], config['single_state_space'], config['local_reward_signal'], env.TL_list, config['batch_size'], config['n_epochs'],
                config['policy_clip'], config['gamma'], config['gae_lambda'], config['policy_learning_rate'], config['value_learning_rate']
                )

if config['is_train']:
    path = set_train_path(config['models_path_name'])
    print('Training results will be saved in:', path)
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
else:
    test_path, plot_path = set_test_path(config['test_model_path_name'])
    print('Test results will be saved in:', plot_path)

# Define the SADQN or MADQN RL training loop
def DQNRL(n_episodes=config['total_episodes'], max_t=config['max_steps'] + 1000, eps_start=config['eps_start'],
          eps_end=config['eps_end'], eps_decay=config['eps_decay'], single_agent=config['single_agent']):
    """Deep Q-Learning (Single-Agent and Multi-Agent).

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        single_agent (bool): whether to use single-agent or multi-agent implementation
    """

    timestamp_start = datetime.datetime.now()
    training_time = []
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    episode_start = 1 if config['is_train'] else n_episodes + 1
    episode_end = n_episodes + 1 if config['is_train'] else n_episodes + 11
    if not config['is_train']:
        agent.load_models(os.path.join(test_path, 'dqn' if single_agent else 'madqn'))
    for i_episode in range(episode_start, episode_end):
        env._TrafficGen.generate_routefile(model_path=env.model_path, model_id=env.model_id, seed=i_episode)
        if not config['is_train']:
            vehicle_stats.create_stats(i_episode)
        state = env.reset()
        score = 0
        for t in range(max_t):
            if config['is_train']:
                q_values, action = agent.act(np.array(state), eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
            else:
                q_values, action = agent.act(np.array(state))
                next_state, reward, done, _ = env.step(action)
            state = next_state
            score += sum(list(reward.values()))
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        training_time.append((datetime.datetime.now() - timestamp_start).total_seconds())
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\r                                       Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(
            scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if not config['is_train']:
            vehicle_stats.store_current_stats(plot_path)
    if config['is_train']:
        if not os.path.exists(os.path.join(path, 'dqn' if single_agent else 'madqn')):
            os.makedirs(os.path.join(path, 'dqn' if single_agent else 'madqn'))
        agent.save_models(os.path.join(path, 'dqn' if single_agent else 'madqn'))
    env.reset()
    env.close()
    if not config['is_train']:
        vehicle_stats.create_overview(plot_path)
    return scores, training_time


# Define the SAPPO or MAPPO RL training loop
def PPORL(n_episodes=config['total_episodes'], max_t=config['max_steps'] + 1000, single_agent=config['single_agent']):
    """Proximal Policy Optimization (Single-Agent and Multi-Agent).

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        single_agent (bool): whether to use single-agent or multi-agent implementation
    """

    best_score = -100000
    learn_every = 64
    timestamp_start = datetime.datetime.now()
    training_time = []
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    episode_start = 1 if config['is_train'] else n_episodes + 1
    episode_end = n_episodes + 1 if config['is_train'] else n_episodes + 11
    if not config['is_train']:
        agent.load_models(os.path.join(test_path, 'ppo' if single_agent else 'mappo'))
    for i_episode in range(episode_start, episode_end):
        env._TrafficGen.generate_routefile(model_path=env.model_path, model_id=env.model_id, seed=i_episode)
        if not config['is_train']:
            vehicle_stats.create_stats(i_episode)
        state = env.reset()
        score = 0
        for t in range(max_t):
            action, prob, val = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            if config['is_train']:
                agent.remember(state, action, prob, val, reward, done)
                if t % learn_every == 0:
                    agent.learn()
            state = state_
            score += sum(list(reward.values()))
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        training_time.append((datetime.datetime.now() - timestamp_start).total_seconds())
        if score > best_score:
            best_score = score
        print('\r                                       Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(
            scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if not config['is_train']:
            vehicle_stats.store_current_stats(plot_path)
    if config['is_train']:
        if not os.path.exists(os.path.join(path, 'ppo' if single_agent else 'mappo')):
            os.makedirs(os.path.join(path, 'ppo' if single_agent else 'mappo'))
        agent.save_models(os.path.join(path, 'ppo' if single_agent else 'mappo'))
    env.reset()
    env.close()
    if not config['is_train']:
        vehicle_stats.create_overview(plot_path)
    return scores, training_time


# Define the WOLP RL training loop
def WOLPRL(n_episodes=config['total_episodes'], max_t=config['max_steps'] + 1000, warmup=config['warmup']):
    """Wolpertinger with Deep Deterministic Policy Gradient.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        warmup (int): time without training but only filling the replay memory and for random action selection
    """

    update_every = 64
    timestamp_start = datetime.datetime.now()
    training_time = []
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    episode_start = 1 if config['is_train'] else n_episodes + 1
    episode_end = n_episodes + 1 if config['is_train'] else n_episodes + 11
    if not config['is_train']:
        agent.load_models(os.path.join(test_path, 'wolp'))
    for i_episode in range(episode_start, episode_end):
        env._TrafficGen.generate_routefile(model_path=env.model_path, model_id=env.model_id, seed=i_episode)
        if not config['is_train']:
            vehicle_stats.create_stats(i_episode)
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            if agent.t_step > warmup * 10 or not config['is_train']:
                raw_action, action = agent.select_action(np.array(state))
            else:
                raw_action, action = agent.random_action()
            action = action.reshape(1,).astype(int)[0]
            next_state, reward, done, _ = env.step(action)
            if config['is_train']:
                agent.observe(state, raw_action, reward, next_state, done)
                agent.step()
                if agent.t_step >= warmup and agent.t_step % update_every == 0:
                    # for _ in range(update_every):
                    agent.update_policy()
            state = next_state
            score += sum(list(reward.values()))
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        training_time.append((datetime.datetime.now() - timestamp_start).total_seconds())
        print('\r                                       Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(
            scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if not config['is_train']:
            vehicle_stats.store_current_stats(plot_path)
    if config['is_train']:
        if not os.path.exists(os.path.join(path, 'wolp')):
            os.makedirs(os.path.join(path, 'wolp'))
        agent.save_models(os.path.join(path, 'wolp'))
    env.reset()
    env.close()
    if not config['is_train']:
        vehicle_stats.create_overview(plot_path)
    return scores, training_time


# Run the training
if 'DQN' in config['agent_type']:
    scores, training_time = DQNRL()

if 'PPO' in config['agent_type']:
    scores, training_time = PPORL()

if config['agent_type'] == 'WOLP':
    scores, training_time = WOLPRL()

print(f'Config: {config}')
print(f'State shape: {env.num_states}')
print(f'Number of actions: {env.action_space.n}')
print(f'Model ID: {env.model_id}')
print(f'Scores: {scores}')
if config['total_episodes'] >= 50 and config['is_train']:
    print(f'Smallest scores (n=50): {heapq.nlargest(50, scores)}')
# print(env.waiting_time)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(os.path.join(path if config['is_train'] else plot_path, 'training_reward.png'))
# plt.show()
DataFrame(data={"reward":scores}).to_csv(os.path.join(path if config['is_train'] else plot_path, 'reward.csv'), sep=',')
DataFrame(data={"reward":scores, "training_time":training_time,
               "waiting_time":env.waiting_time["TL"],"queue_length":env.queue["TL"]}
         ).to_csv(os.path.join(path if config['is_train'] else plot_path, 'training_stats.csv'), sep=',', index=False)
add_masterdata(path if config['is_train'] else plot_path, config, scores, training_time, env.waiting_time["TL"], env.waiting_time["TL"])
