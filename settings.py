import os
import sys
# from types import SimpleNamespace

import configparser
import argparse

_settings = {
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


def _import_configuration():
    """Read the appropriate config file (for training or testing)."""
    config_file = 'training_settings.ini'
    options = _get_cli_options()
    print(options)
    config = _parse_config_file(config_file, is_train_config=True)
    _overwrite_config_with_options(config, options)

    if not config['is_train']:
        test_config_file = os.path.join(config['test_model_path_name'], config_file)
        config = _parse_config_file(test_config_file, is_train_config=False)
        _overwrite_config_with_options(config, options)

    return config


def _parse_config_file(config_file, is_train_config):

    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    for category, category_settings in _settings.items():
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


def _get_cli_options():
    converters = {
        'int': int,
        'float': float,
        'string': str,
    }

    arg_parser = argparse.ArgumentParser()

    for category, category_settings in _settings.items():
        for setting, setting_type in category_settings.items():
            cli_option = setting.replace('_', '-')

            if setting_type == 'bool':
                action = 'store_true'
                arg_parser.add_argument(f'--{cli_option}', action=action, default=None, dest=setting)
                arg_parser.add_argument(f'--not-{cli_option}', action='store_false', default=None, dest=setting)
            elif setting_type == 'int_list':
                arg_parser.add_argument(f'--{cli_option}', action=_IntListAction, default=None, dest=setting)
            else:
                action = 'store'
                convert_type = converters[setting_type]
                arg_parser.add_argument(f'--{cli_option}', action=action, type=convert_type, default=None, dest=setting)

    arg_parser.add_argument('--test', action='store_false', dest='is_train')
    return arg_parser.parse_args()


class _IntListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        value = [int(v) for v in values.split(',')]
        setattr(namespace, self.dest, value)


def _overwrite_config_with_options(config, options):
    for option_name, value in options.__dict__.items():
        if value is not None:
            config[option_name] = value


# config = SimpleNamespace(**_import_configuration())
config = _import_configuration()
