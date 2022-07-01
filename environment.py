import os
import random
import sys
from itertools import product

import numpy as np
import traci
from gym import Env
from gym.spaces import Discrete
from sumolib import checkBinary

from stats.junction import JunctionStatistics, TL_list
from traffic_generation import ModularTrafficGenerator
from modular_road_network_structure import create_modular_road_network
from settings import config
from state import JunctionManager


class SUMO(Env):
    def __init__(self, vehicle_stats=None):
        import_sumo_tools()

        self.vehicle_stats = vehicle_stats

        # Create the road topology
        self.model_path, self.model_id = create_modular_road_network(
            config['models_path_name'], config['num_intersections'], config['intersection_length']
        )

        # Create a generator for creating traffic routes
        self._traffic_generator = ModularTrafficGenerator(
            config['max_steps'],
            config['n_cars_generated'],
            os.path.join(self.model_path, 'environment.net.xml')
        )

        self.traffic_lights = {}

        for i in range(1, config['num_intersections'] + 1):
            intersection_name = f'TL{i}'
            self.traffic_lights[intersection_name] = str(i)

        self._sumo_cmd = configure_sumo(config['gui'], self.model_path, config['sumocfg_file_name'], config['max_steps'])
        self.num_intersections = config['num_intersections']

        if config['single_agent']:
            if config['fixed_action_space']:
                self.action_space = Discrete(config['num_actions'])
            else:
                self.single_action_space = Discrete(config['num_actions'])
                self.action_space = Discrete(pow(config['num_actions'], self.num_intersections))
                # self.action_space_combinations = list(product(range(0, self.single_action_space.n), repeat=self.num_intersections))
        else:
            self.action_space = Discrete(config['num_actions'])
            self.num_agents = len(self.traffic_lights)

        self.green_duration = config['green_duration']
        self.yellow_duration = config['yellow_duration']
        self.red_duration = config['red_duration']

        self.generate_traffic()
        traci.start(self._sumo_cmd)

        self.junction_manager = JunctionManager(self.traffic_lights)
        self.num_states = sum(i + 1 for i in list(self.junction_manager.get_state_sizes().values()))
        self.init_state = [0] * self.num_states
        self.state = [0] * self.num_states

        self.junction_stats = JunctionStatistics(self.traffic_lights)

        self.waiting_time = dict.fromkeys(TL_list)
        self.queue = dict.fromkeys(TL_list)

        self.old_actions = dict.fromkeys(self.traffic_lights, 0)
        self.reward = 0

    def generate_traffic(self, seed=random.randint(0, 100)):
        self._traffic_generator.generate_route_file(model_path=self.model_path, seed=seed)

    def _compute_action_combination(self, actions):
        # The action combination can be viewed as converting the actions index (in base 10) to a number in base N,
        # where N is the number of possible actions, and each digit of this result is then the action of a single
        # intersection. This result must also be padded with leading zeros so that the number of digits is equal
        # to the number of intersections.
        base = self.single_action_space.n
        num_digits = self.num_intersections

        action_combination = []

        while actions:
            action_combination.append(actions % base)
            actions //= base

        while len(action_combination) < num_digits:
            action_combination.append(0)

        return list(reversed(action_combination))

    def _fixed_action_generator(self, actions):
        for tl_id in actions:
            yield self.old_actions[tl_id], actions[tl_id], tl_id

    def _multi_action_generator(self, actions):
        # action_combination = self.action_space_combinations[actions]
        action_combination = self._compute_action_combination(actions)

        for tl_index, tl_id in enumerate(self.traffic_lights):
            yield self.old_actions[tl_id], action_combination[tl_index], tl_id

    def step(self, actions: dict):
        if config['fixed_action_space'] or config['single_agent'] is False:
            action_generator = self._fixed_action_generator
        else:
            action_generator = self._multi_action_generator

        for (old_action, new_action, traffic_light_id) in action_generator(actions):
            if old_action != new_action:
                self._set_yellow_phase(old_action, traffic_light_id)
            else:
                self._set_green_phase(new_action, traffic_light_id)

        self._simulate(self.yellow_duration)

        for (old_action, new_action, traffic_light_id) in action_generator(actions):
            if old_action != new_action:
                self._set_red_phase(old_action, traffic_light_id)

        self._simulate(self.red_duration)

        for (old_action, new_action, traffic_light_id) in action_generator(actions):
            self._set_green_phase(new_action, traffic_light_id)
            self.old_actions[traffic_light_id] = new_action

        self._simulate(self.green_duration)

        # Removed because it slows training down
        # self.junction_stats.step()

        if config['reward_definition'] == 'waiting':
            self.junction_manager.receive_rewards()

        self.reward = self.junction_manager.compute_rewards()

        full_state = self.junction_manager.return_states()
        self.state = [item for sublist in list(full_state.values()) for item in sublist]

        if len(self.state) == 0:
            self.state = self.init_state

        if traci.simulation.getTime() >= config['max_steps'] + 1000:
            done = True
            traci.close()
        else:
            done = False

        info = {}
        return self.state, self.reward, done, info

    def _simulate(self, steps_todo):
        for i in range(steps_todo):
            traci.simulationStep()

            # Removed because it slows training down
            # self.junction_stats.simulate()

            if config['reward_definition'] == 'waiting_fast':
                self.junction_manager.receive_rewards()

            if self.vehicle_stats is not None:
                self.vehicle_stats.update()

            self.junction_manager.receive_states()

    @staticmethod
    def _set_green_phase(action_number, tl_id):
        """
        Activate the correct green light combination in sumo
        """
        green_phase_code = action_number * 3  # Obtain the green phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, green_phase_code)

    @staticmethod
    def _set_yellow_phase(old_action, tl_id):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 3 + 1  # Obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, yellow_phase_code)

    @staticmethod
    def _set_red_phase(old_action, tl_id):
        """
        Activate the correct red light combination in sumo
        """
        red_phase_code = old_action * 3 + 2  # Obtain the red phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, red_phase_code)

    def reset(self):
        self._sumo_cmd = configure_sumo(config['gui'], self.model_path, config['sumocfg_file_name'], config['max_steps'])

        try:
            traci.start(self._sumo_cmd)
        except:
            traci.close()
            traci.start(self._sumo_cmd)

        self.state = [0] * self.num_states
        self.waiting_time, self.queue = self.junction_stats.compute_means(self.waiting_time, self.queue)
        return self.state

    def render(self):
        # Implement visualization
        pass


def import_sumo_tools():
    """
    Import Python modules from the $SUMO_HOME/tools directory.
    """
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")


def configure_sumo(gui, model_path, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO.
    """
    # Setting the cmd mode or the visual mode
    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    # Setting the cmd command to run sumo at simulation time
    model_path = os.path.join(model_path, sumocfg_file_name)
    sumo_cmd = [sumo_binary, "-c", model_path, "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd
