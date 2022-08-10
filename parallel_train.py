import itertools
import multiprocessing
import time

PROCESS_INTERVAL = 10


def train_process(settings_combination):
    from settings import config
    config.overwrite(settings_combination)
    print(settings_combination)

    # Import statement must be after config is updated with settings_combination,
    # otherwise the training will receive the default training settings
    from train import main
    main()


def train_in_parallel():
    setting_changes = {
        'green_duration': [1, 10],
        'red_duration': [3, 5],
        'total_episodes': [2]
    }

    settings, values = zip(*setting_changes.items())
    setting_combinations = [dict(zip(settings, value_combo)) for value_combo in itertools.product(*values)]

    processes = [
        multiprocessing.Process(target=train_process, args=(settings_combination,))
        for settings_combination in setting_combinations
    ]

    for process, setting_combination in zip(processes, setting_combinations):
        process.start()
        time.sleep(PROCESS_INTERVAL)

    for process in processes:
        process.join()


if __name__ == '__main__':
    train_in_parallel()

