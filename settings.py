import os
import sys
from enum import Enum, EnumMeta, auto
# from types import SimpleNamespace

import configparser
import argparse


class TrainingStrategy(Enum):
    NONSTRATEGIC = auto()
    CONCURRENT = auto()
    CENTRALIZED = auto()


class RewardDefinition(Enum):
    WAITING = auto()
    WAITING_FAST = auto()


class StateRepresentation(Enum):
    VOLUME_LANE = auto()
    VOLUME_LANE_FAST = auto()
    WAITING_T = auto()
    STAULANGE = auto()


class AgentType(Enum):
    DQN = auto()
    MADQN = auto()
    PPO = auto()
    MAPPO = auto()
    RAINBOW_DQN = auto()
    WOLP = auto()

    def is_multi(self):
        return self == self.MADQN or self == self.MAPPO


_settings = {
        'simulation': {
            'gui': bool,
            'total_episodes': int,
            'max_steps': int,
            'n_cars_generated': int,
            'generation_process': str,
            'green_duration': int,
            'yellow_duration': int,
            'red_duration': int,
            'num_intersections': int,
            'intersection_length': int,
            'cycle_time' : int
        },

        'model': {
            'hidden_dim': list,
            'critic_dim': list,
            'actor_dim': list,
            'batch_size': int,
            'learning_rate': float,
            'num_layers': int,
            'policy_learning_rate': float,
            'value_learning_rate': float,
            'actor_init_w': float,
            'critic_init_w': float,
            'weight_decay': float,
            'training_epochs': int,
            'target_update': int,
            'warmup': int,
        },

        'memory': {
            'memory_size_min': int,
            'memory_size_max': int,
        },

        'strategy': {
            'eps_start': float,
            'eps_end': float,
            'eps_decay': float,
            'eps_policy': int,
        },

        'agent': {
            'agent_type': AgentType,
            'model': str,
            'is_train': bool,
            'state_representation': StateRepresentation,
            'action_representation': str,
            'reward_definition': RewardDefinition,
            'training_strategy': TrainingStrategy,
            'actor_parameter_sharing': bool,
            'critic_parameter_sharing': bool,
            'num_states': int,
            'num_actions': int,
            'action_definition': str,
            'single_state_space': bool,
            'fixed_action_space': bool,
            'local_reward_signal': bool,
            'gamma': float,
            'tau': float,
            'ou_theta': float,
            'ou_mu': float,
            'ou_sigma': float,
            'gae_lambda': float,
            'policy_clip': float,
            'n_epochs': int,
            'learning_interval': int,
        },

        'dir': {
            'models_path_name': str,
            'test_model_path_name': str,
            'sumocfg_file_name': str,
        }
    }


def _import_configuration():
    """Read the appropriate config file (for training or testing)."""
    config_file = 'training_settings.ini'
    options = _get_cli_options()
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
            if setting_type == bool:
                value = content[category].getboolean(setting)
            elif setting_type == int:
                value = content[category].getint(setting)
            elif setting_type == float:
                value = content[category].getfloat(setting)
            elif setting_type == str:
                value = content[category].get(setting)
            elif setting_type == list:
                value = [int(v) for v in content[category].get(setting).split(',')]
            elif type(setting_type) == EnumMeta:
                value_str = content[category].get(setting)

                try:
                    value = setting_type[value_str.upper()]
                except KeyError:
                    sys.exit(
                        f'Invalid value "{value_str}" for setting "{setting}". '
                        f'Valid values are: {[name.lower() for name in setting_type.__members__]}'
                    )
            else:
                sys.exit(f'Invalid type "{setting_type}" for setting "{setting}"')

            config[setting] = value

    # Handle the multi-agent and single agent cases
    if config['agent_type'].is_multi():
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


def _create_enum_converter(enum_class):
    def convert_to_enum(arg: str):
        return enum_class[arg.upper()]

    return convert_to_enum


def _get_cli_options():
    arg_parser = argparse.ArgumentParser()

    for category, category_settings in _settings.items():
        for setting, setting_type in category_settings.items():
            cli_option = setting.replace('_', '-')

            if setting_type == bool:
                action = 'store_true'
                arg_parser.add_argument(f'--{cli_option}', action=action, default=None, dest=setting)
                arg_parser.add_argument(f'--not-{cli_option}', action='store_false', default=None, dest=setting)
            elif setting_type == list:
                arg_parser.add_argument(f'--{cli_option}', action=_IntListAction, default=None, dest=setting)
            else:
                action = 'store'
                arg_type = _create_enum_converter(setting_type) if type(setting_type) == EnumMeta else setting_type
                arg_parser.add_argument(f'--{cli_option}', action=action, type=arg_type, default=None, dest=setting)

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
