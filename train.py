import heapq
import os
import sys
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from environment import SumoPhaseEnv, SumoCycleEnv
from training import DQNTraining, PPOTraining, WOLPTraining
from settings import config, AgentType, ActionDefinition
from utils import logger, create_train_path, create_test_path, add_master_data

is_train = config['is_train']

# Set up the folder for the training/test results
if is_train:
    path, _ = create_train_path(config['models_path_name'])
    print('Training results will be saved in:', path)
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
else:
    path, plot_path = create_test_path(config['test_model_path_name'])
    print('Test results will be saved in:', plot_path)

logger.set_log_file(os.path.join(path, 'log.txt'))

logger.log('Training settings')
for setting, value in config.items():
    logger.log(f'{setting}: {value}')
logger.log('\n')

# Set up the corresponding SUMO environment for training or testing
if is_train:
    vehicle_stats = None
else:
    from stats.vehicle import VehicleStatistics
    vehicle_stats = VehicleStatistics()

if config['action_definition'] == ActionDefinition.PHASE:
    env = SumoPhaseEnv(vehicle_stats)
    action_size = env.action_space.n
elif config['action_definition'] == ActionDefinition.CYCLE:
    env = SumoCycleEnv(vehicle_stats)
    action_size = env.action_space
else:
    raise ValueError('No SUMO environment specified for the selected action_definition')

logger.log('Number of actions: ', action_size)
logger.log('State shape: ', env.num_states)

# Create agent based on config file, and train it
agent_type = config['agent_type']
if agent_type == AgentType.RAINBOW_DQN:
    from agents.rainbow_dqn_agent import RainbowDQNAgent
    agent = RainbowDQNAgent(
        env.num_states, action_size, config['hidden_dim'], config['fixed_action_space'],
        env.traffic_lights, config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'],
        config['learning_rate'], config['target_update'], True, 1, True, True
    )
    scores, training_times = DQNTraining(agent, env, 'dqn').train(is_train=is_train)

elif agent_type == AgentType.DQN:
    from agents.dqn_agent import DQNAgent
    agent = DQNAgent(
        env.num_states, action_size, config['hidden_dim'], config['fixed_action_space'],
        env.traffic_lights, config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'],
        config['learning_rate'], config['target_update'],
    )
    scores, training_times = DQNTraining(agent, env, 'dqn').train(is_train=is_train)

elif agent_type == AgentType.PPO:
    from agents.ppo_agent import PPOAgent
    agent = PPOAgent(
        env.num_states, action_size, config['actor_dim'], config['critic_dim'], config['fixed_action_space'],
        env.traffic_lights, config['batch_size'], config['n_epochs'], config['policy_clip'], config['gamma'],
        config['gae_lambda'], config['policy_learning_rate'], config['value_learning_rate'],
        config['learning_interval'], config['action_definition']
    )
    scores, training_times = PPOTraining(agent, env, 'ppo').train(is_train=is_train)

elif agent_type == AgentType.WOLP:
    from deprecated.wolp_agent import WolpertingerAgent
    agent = WolpertingerAgent(
        env.num_states, action_size, config['actor_dim'], config['critic_dim'], config['eps_policy'],
        config['actor_init_w'], config['critic_init_w'], config['memory_size_max'], config['batch_size'],
        config['gamma'], config['tau'], config['ou_theta'], config['ou_mu'], config['ou_sigma'],
        config['policy_learning_rate'], config['value_learning_rate'], config['weight_decay']
    )
    scores, training_times = WOLPTraining(agent, env, 'wolp').train(is_train=is_train)

elif agent_type == AgentType.MADQN:
    from agents.madqn_agent import MADQNAgent
    agent = MADQNAgent(
        env.num_states, action_size, len(env.traffic_lights), config['hidden_dim'], config['single_state_space'],
        config['local_reward_signal'],
        env.traffic_lights, config['memory_size_max'], config['batch_size'], config['gamma'], config['tau'],
        config['learning_rate'], config['target_update']
    )
    scores, training_times = DQNTraining(agent, env, 'madqn').train(is_train=is_train)

elif agent_type == AgentType.MAPPO:
    from agents.mappo_agent import MAPPOAgent
    agent = MAPPOAgent(
        env.num_states, action_size, len(env.traffic_lights), config['actor_dim'], config['critic_dim'],
        config['training_strategy'], config['actor_parameter_sharing'], config['critic_parameter_sharing'],
        config['single_state_space'], config['local_reward_signal'],env.traffic_lights, config['batch_size'],
        config['n_epochs'], config['policy_clip'], config['gamma'], config['gae_lambda'],
        config['policy_learning_rate'], config['value_learning_rate']
    )
    scores, training_times = PPOTraining(agent, env, 'mappo').train(is_train=is_train)

else:
    sys.exit(f'Invalid agent_type: {agent_type} is not an implemented agent')

# Log model information
logger.log(f'Config: {config}')
logger.log(f'State shape: {env.num_states}')
logger.log(f'Number of actions: {action_size}')
logger.log(f'Model ID: {env.model_id}')
logger.log(f'Scores: {scores}')
BEST_SCORE_WINDOW = 50
if config['total_episodes'] >= BEST_SCORE_WINDOW and is_train:
    logger.log(f'Smallest scores (n={BEST_SCORE_WINDOW}): {heapq.nlargest(BEST_SCORE_WINDOW, scores)}')

data_path = path if is_train else plot_path

# Plot the scores from training or testing
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(os.path.join(data_path, 'training_reward.png'))
# plt.show()

# Save the results from training or testing
DataFrame(data={'reward': scores}).to_csv(os.path.join(data_path, 'reward.csv'), sep=',')
DataFrame(data={
    'reward': scores,
    'training_time': training_times,
    'waiting_time': env.waiting_time['TL'],
    'queue_length': env.queue['TL']
}).to_csv(os.path.join(data_path, 'training_stats.csv'), sep=',', index=False)

add_master_data(data_path, config, scores, training_times, env.waiting_time['TL'], env.waiting_time['TL'])
