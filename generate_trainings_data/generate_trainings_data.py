import numpy as np
import datetime
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

import os


trainings_results = [
    {
        'model_id': '',
        'config': {},
        'scores': []
    },
]


def set_trainings_data_path(algorithm, num_intersections, hidden_dim):
    trainings_data_path = os.path.join(os.getcwd(), 'generate_trainings_data', '')

    data_path = os.path.join(trainings_data_path, f'{algorithm}_{num_intersections}i_{hidden_dim}', '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    return data_path

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
                      'training_time':training_time,
                      'final_waiting_time':wait,
                      'final_length':queue}, ignore_index=True)
    master_df.to_excel('Masterdata.xlsx', index=False)

def generate_trainings_data(config, scores):
    algorithm = config['agent_type'].lower()
    algorithm = algorithm if config['fixed_action_space'] is False else f'{algorithm}fixed'
    num_intersections = config['num_intersections']
    hidden_dim = config['hidden_dim'][1:-1]
    hidden_dim = ','.join(hidden_dim)
    path = set_trainings_data_path(algorithm, num_intersections, hidden_dim)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(path, 'training_reward.png'))

    DataFrame(data={"reward": scores}).to_csv(os.path.join(path, 'reward.csv'), sep=',')
    DataFrame(data={"reward": scores, "training_time": '', "waiting_time": '', "queue_length": ''}
             ).to_csv(os.path.join(path, 'training_stats.csv'), sep=',', index=False)
    add_masterdata(path, config, scores, '', '', '')


if __name__ == '__main__':
    for training_res in trainings_results:
        generate_trainings_data(training_res['config'], training_res['scores'])
