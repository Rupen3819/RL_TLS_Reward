import os
import sys
from types import SimpleNamespace

import configparser


def _import_configuration():
    """Read the appropriate config file (for training or testing)."""
    config_file = 'training_settings.ini'
    config = _parse_config_file(config_file, is_train_config=True)

    if not config['is_train']:
        test_config_file = os.path.join(config['test_model_path_name'], config_file)
        config = _parse_config_file(test_config_file, is_train_config=False)

    return config


def _parse_config_file(config_file, is_train_config):
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
            'num_intersections': 'int',
            'intersection_length': 'int',
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
        config['is_train'] = False

    return config


# config = SimpleNamespace(**_import_configuration())
config = _import_configuration()
