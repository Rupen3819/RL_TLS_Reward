import datetime
import os
from shutil import copyfile
from collections import deque

import numpy as np

from environment import SumoEnv
from settings import config
from utils import create_train_path, create_test_path


class EpisodicTraining:
    TEST_EPISODES = 10
    EPISODE_INTERVAL = 100

    def __init__(self, agent, env: SumoEnv, model_name: str, n_episodes: int, max_t: int):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.model_name = model_name

    def run_episode(self, state, is_train: bool) -> int:
        raise NotImplementedError('Must implement training "step" method')

    def train(self, is_train: bool = True) -> tuple[list[int], list[float]]:
        timestamp_start = datetime.datetime.now()
        training_times = []
        scores = []  # Contains all episode scores
        scores_window = deque(maxlen=self.EPISODE_INTERVAL)  # Recent window of episode scores

        if is_train:
            path, _ = create_train_path(config['models_path_name'])
            model_path = os.path.join(path, self.model_name)
            print('Training results will be saved in:', path)
            copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
            episode_start = 1
            episode_end = self.n_episodes + 1
        else:
            test_path, plot_path = create_test_path(config['test_model_path_name'])
            test_model_path = os.path.join(test_path, self.model_name)
            print('Test results will be saved in:', plot_path)
            episode_start = self.n_episodes + 1
            episode_end = self.n_episodes + self.TEST_EPISODES + 1
            self.agent.load_model(test_model_path)

        for i_episode in range(episode_start, episode_end):
            self.env.generate_traffic(seed=i_episode)

            if not is_train:
                self.env.vehicle_stats.create(i_episode)

            state = self.env.reset()
            score = self.run_episode(state, is_train)

            # Save most recent score
            scores_window.append(score)
            scores.append(score)

            training_times.append((datetime.datetime.now() - timestamp_start).total_seconds())
            print(
                '\r                                       Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(
                    scores_window)), end="")

            if i_episode % self.EPISODE_INTERVAL == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if not is_train:
                self.env.vehicle_stats.save(plot_path)

        if is_train and not os.path.exists(model_path):
            os.makedirs(model_path)
            self.agent.save_model(model_path)

        self.env.reset()
        self.env.close()

        if not is_train:
            self.env.vehicle_stats.generate_summary(plot_path)

        return scores, training_times


class DQNTraining(EpisodicTraining):
    def __init__(
            self,
            agent,
            env: SumoEnv,
            model_name: str,
            n_episodes: int = config['total_episodes'],
            max_t: int = config['max_steps'] + 1000,
            eps_start: float = config['eps_start'],
            eps_end: float = config['eps_end'],
            eps_decay: float = config['eps_decay'],
    ):
        super().__init__(agent, env, model_name, n_episodes, max_t)
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def run_episode(self, state, is_train: bool = True) -> int:
        score = 0

        for t in range(self.max_t):
            if is_train:
                q_values, action = self.agent.act(np.array(state), self.eps)
                # action = [16,16,17,17] #Delete for normal case
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
            else:
                q_values, action = self.agent.act(np.array(state))
                next_state, reward, done, _ = self.env.step(action)

            state = next_state
            score += sum(list(reward.values()))

            if done:
                break

        self.eps = max(self.eps_end, self.eps_decay * self.eps)  # Decrease epsilon

        return score


class PPOTraining(EpisodicTraining):
    def __init__(
            self,
            agent,
            env: SumoEnv,
            model_name: str,
            n_episodes: int = config['total_episodes'],
            max_t: int = config['max_steps'] + 1000,
    ):
        super().__init__(agent, env, model_name, n_episodes, max_t)

    def run_episode(self, state, is_train: bool = True) -> int:
        score = 0

        for t in range(self.max_t):
            action, value, log_prob = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)

            if is_train:
                self.agent.step(state, action, reward, done, value, log_prob)

            state = next_state
            score += sum(reward.values())

            if done:
                break

        return score


class WOLPTraining(EpisodicTraining):
    UPDATE_EVERY = 64

    def __init__(
            self,
            agent,
            env: SumoEnv,
            model_name: str,
            n_episodes: int = config['total_episodes'],
            max_t: int = config['max_steps'] + 1000,
            warmup: int = config['warmup']
    ):
        super().__init__(agent, env, model_name, n_episodes, max_t)
        self.warmup = warmup

    def run_episode(self, state, is_train: bool = True) -> int:
        score = 0
        for t in range(self.max_t):
            if self.agent.t_step > self.warmup * 10 or not is_train:
                raw_action, action = self.agent.select_action(np.array(state))
            else:
                raw_action, action = self.agent.random_action()

            action = action.reshape(1, ).astype(int)[0]
            next_state, reward, done, _ = self.env.step(action)

            if is_train:
                self.agent.observe(state, raw_action, reward, next_state, done)
                self.agent.step()

                if self.agent.t_step >= self.warmup and self.agent.t_step % self.UPDATE_EVERY == 0:
                    self.agent.update_policy()

            state = next_state
            score += sum(list(reward.values()))
            if done:
                break

        return score
