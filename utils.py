import datetime
import os
import sys

import pandas as pd

MASTER_DATA_FILE = 'Masterdata.xlsx'


def set_intersection_path(models_path_name):
    """
    Create a new intersection model path with an incremental integer, also considering previously created model paths.
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


def create_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    new_version = 1
    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split('_')[-1]) for name in dir_content if not name.startswith('.')]
        new_version = max(previous_versions) + 1

    data_path = os.path.join(models_path, 'model_' + str(new_version), '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def create_test_path(test_model_path_name):
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


def add_master_data(path, config, scores, training_time, wait, queue):
    master_df = pd.read_excel(MASTER_DATA_FILE)
    path = path[0:-1]
    name = os.path.split(path)[1]
    master_df = pd.concat([master_df, pd.DataFrame([{
        'run_name': name,
        'datetime': datetime.datetime.now(),
        'agent_type': config['agent_type'],
        'model': config['model'],
        'total_episodes': config['total_episodes'],
        'generation_process': config['agent_type'],
        'num_states': config['num_states'],
        'cars_generated': config['n_cars_generated'],
        'num_actions': config['num_actions'],
        'state_representation': config['state_representation'],
        'action_representation': config['action_representation'],
        'final_reward': scores[-1],
        'training_time': training_time[-1],
        'final_waiting_time': wait,
        'final_length': queue
    }])], ignore_index=True)

    master_df.to_excel(MASTER_DATA_FILE, index=False)
