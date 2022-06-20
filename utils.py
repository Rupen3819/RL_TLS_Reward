import os
import sys
import configparser
from sumolib import checkBinary
import pandas as pd
import datetime

import torch
from torch.autograd import Variable


# Import all settings from the config file for training or testing
def import_configuration():
    """Read the appropriate config file (for training or testing)."""
    config_file = 'training_settings.ini'
    config = parse_config_file(config_file, is_train_config=True)

    if not config['is_train']:
        test_config_file = os.path.join(config_file, config['test_model_path_name'])
        config = parse_config_file(test_config_file, is_train_config=False)

    return config


def parse_config_file(config_file, is_train_config):
    settings = {
        'simulation': {
            'gui': 'bool',
            'total_episodes': 'int',
            'max_steps': 'int',
            'n_cars_generated': 'int',
            'generation_process': 'string',
            'green_duration': 'int',
            'yellow_duration': 'int',
            'red_duration': 'int',
            'num_intersections': 'string',
            'intersection_length': 'string',
        },

        'model': {
            'hidden_dim': 'int_list',
            'critic_dim': 'int_list',
            'actor_dim': 'int_list',
            'batch_size': 'int',
            'learning_rate': 'float',
            'num_layers': 'int',
            'policy_learning_rate': 'float',
            'value_learning_rate': 'float',
            'actor_init_w': 'float',
            'critic_init_w': 'float',
            'weight_decay': 'float',
            'training_epochs': 'int',
            'target_update': 'int',
            'warmup': 'int',
            # 'width_layers': 'int',
        },

        'memory': {
            'memory_size_min': 'int',
            'memory_size_max': 'int',
        },

        'strategy': {
            'eps_start': 'float',
            'eps_end': 'float',
            'eps_decay': 'float',
            'eps_policy': 'int',
        },

        'agent': {
            'agent_type': 'string',
            'model': 'string',
            'is_train': 'bool',
            'state_representation': 'string',
            'action_representation': 'string',
            'reward_definition': 'string',
            'training_strategy': 'string',
            'actor_parameter_sharing': 'bool',
            'critic_parameter_sharing': 'bool',
            'num_states': 'int',
            'num_actions': 'int',
            'single_state_space': 'bool',
            'fixed_action_space': 'bool',
            'local_reward_signal': 'bool',
            'gamma': 'float',
            'tau': 'float',
            'ou_theta': 'float',
            'ou_mu': 'float',
            'ou_sigma': 'float',
            'gae_lambda': 'float',
            'policy_clip': 'float',
            'n_epochs': 'int',
        },

        'dir': {
            'models_path_name': 'string',
            'test_model_path_name': 'string',
            'sumocfg_file_name': 'string',
        }
    }

    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    for category, category_settings in settings.items():
        for setting, setting_type in category_settings.items():
            match setting_type:
                case 'bool':
                    value = content[category].getboolean(setting)
                case 'int':
                    value = content[category].getint(setting)
                case 'float':
                    value = content[category].getfloat(setting)
                case 'string':
                    value = content[category].get(setting)
                case 'int_list':
                    value = [int(v) for v in content[category].get(setting).split(',')]
                case _:
                    sys.exit(f'Invalid type "{setting_type}" for setting "{setting}"')

            config[setting] = value

    # Handle the multi-agent and single agent cases
    if config['agent_type'].startswith('MA'):
        config['single_agent'] = False
        config['fixed_action_space'] = False
    else:
        config['single_agent'] = True
        config['single_state_space'] = False
        config['local_reward_signal'] = False

    # Change settings for test configuration
    if not is_train_config:
        config['test_model_path_name'] = config_file
        config['is_train'] = False

    return config


def set_sumo(gui, model_path, model_id, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join(f'intersection/{model_path}/model_{model_id}', sumocfg_file_name), "--no-step-log", "true",
                "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


def set_intersection_path(models_path_name):
    """
    Create a new intersection model path with an incremental integer, also considering previously created model paths
    """
    models_path = models_path_name.split('/', 1)[1]
    train_model_path = os.path.join(os.getcwd(), f'models/{models_path}', '')
    intersection_model_path = os.path.join(os.getcwd(), f'intersection/{models_path}', '')
    os.makedirs(os.path.dirname(train_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(intersection_model_path), exist_ok=True)

    dir_content = os.listdir(train_model_path)
    if dir_content:
        for d in dir_content:
            if d == '.DS_Store':
                os.remove(os.path.join(train_model_path, d))
                dir_content = os.listdir(train_model_path)
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(intersection_model_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return new_version


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        for d in dir_content:
            if d == '.DS_Store':
                os.remove(os.path.join(models_path, d))
                dir_content = os.listdir(models_path)
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(test_model_path_name):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), test_model_path_name, '')

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')


def add_masterdata(path, config, scores, training_time, wait, queue):
    master_df = pd.read_excel('Masterdata.xlsx')
    path = path[0:-1]
    name = os.path.split(path)[1]
    master_df = master_df.append({'run_name':name,
                      'datetime':datetime.datetime.now(),
                      'agent_type':config['agent_type'],
                      'model':config['model'],
                      'total_episodes':config['total_episodes'],
                      'generation_process':config['agent_type'],
                      'num_states':config['num_states'],
                      'cars_generated':config['n_cars_generated'],
                      'num_actions':config['num_actions'],
                       'state_representation':config['state_representation'],
                      'action_representation':config['action_representation'],
                      'final_reward':scores[-1],
                      'training_time':training_time[-1],
                    #   'final_waiting_time':wait[-1],
                    #   'final_length':queue[-1],
                      'final_waiting_time':wait,
                      'final_length':queue}, ignore_index=True)
    master_df.to_excel('Masterdata.xlsx', index=False)
