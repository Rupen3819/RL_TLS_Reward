import os
import random
import sys
from itertools import product

import traci
from gym import Env
from gym.spaces import Discrete
from sumolib import checkBinary

from Statistics import init_statistics, return_mean_statistics, TL_list
from generator import ModularTrafficGenerator
from modular_road_network_structure import create_modular_road_network
from settings import config
from state import get_current_reward, return_reward, return_states, init_states, get_states, get_state_size


class SUMO(Env):
    def __init__(self, vehicle_stats=None):
        import_sumo_tools()

        self.vehicle_stats = vehicle_stats
        self.model_path, self.model_id = create_modular_road_network(
            config['models_path_name'], config['num_intersections'], config['intersection_length']
        )

        self._traffic_generator = ModularTrafficGenerator(
            config['max_steps'],
            config['n_cars_generated'],
            f'intersection/{self.model_path}/model_{self.model_id}/environment.net.xml'
        )

        self.TL_list, self.action_dict, self.program_dict, self.num_program_dict = self.get_tl_dicts(config['num_intersections'])

        self._sumo_cmd = configure_sumo(config['gui'], self.model_path, self.model_id, config['sumocfg_file_name'], config['max_steps'])

        if config['single_agent']:
            if config['fixed_action_space']:
                self.action_space = Discrete(config['num_actions'])
            else:
                self.single_action_space = Discrete(config['num_actions'])
                self.action_space = Discrete(pow(config['num_actions'], config['num_intersections']))
                self.action_space_combinations = list(product(list(range(0, self.single_action_space.n)), repeat=config['num_intersections']))
        else:
            self.action_space = Discrete(config['num_actions'])
            self.num_agents = len(self.TL_list)

        self.green_duration = config['green_duration']
        self.yellow_duration = config['yellow_duration']
        self.red_duration = config['red_duration']

        self.generate_traffic()
        traci.start(self._sumo_cmd)

        self.junction_dict = init_states(self.TL_list)
        self.num_states = sum(i + 1 for i in list(get_state_size(self.junction_dict, self.num_program_dict).values()))
        self.init_state = [0] * self.num_states
        self.state = [0] * self.num_states

        self.junction_statistics_dict = init_statistics(self.TL_list)
        self.waiting_time = dict.fromkeys(TL_list)
        self.queue = dict.fromkeys(TL_list)

        self._old_action = dict.fromkeys(self.TL_list, 0)
        self._old_total_wait = 0
        self._waiting_times = {}
        self._waiting_old = 0
        self.reward = 0
        self.reward_old = 0

    def generate_traffic(self, seed=random.randint(0, 100)):
        self._traffic_generator.generate_routefile(model_path=self.model_path, model_id=self.model_id, seed=seed)

    def step(self, action):
        if config['fixed_action_space'] or config['single_agent'] is False:
            for tl_id in action:
                if self._old_action[tl_id] != action[tl_id]:
                    self._set_yellow_phase(self._old_action[tl_id], tl_id)
                else:
                    self._set_green_phase(action[tl_id], tl_id)
            self._simulate(self.yellow_duration)

            for tl_id in action:
                if self._old_action[tl_id] != action[tl_id]:
                    self._set_red_phase(self._old_action[tl_id], tl_id)
            self._simulate(self.red_duration)

            for tl_id in action:
                self._set_green_phase(action[tl_id], tl_id)
                self._simulate(self.green_duration)
                self._old_action[tl_id] = action[tl_id]
        else:
            action_combination = self.action_space_combinations[action]

            for tl_index, tl_id in enumerate(self.TL_list):
                if self._old_action[tl_id] != action_combination[tl_index]:
                    self._set_yellow_phase(self._old_action[tl_id], tl_id)
                else:
                    self._set_green_phase(action_combination[tl_index], tl_id)
            self._simulate(self.yellow_duration)

            for tl_index, tl_id in enumerate(self.TL_list):
                if self._old_action[tl_id] != action_combination[tl_index]:
                    self._set_red_phase(self._old_action[tl_id], tl_id)
            self._simulate(self.red_duration)

            for tl_index, tl_id in enumerate(self.TL_list):
                self._set_green_phase(action_combination[tl_index], tl_id)
                self._simulate(self.green_duration)
                self._old_action[tl_id] = action_combination[tl_index]

        # Removed because it slows training down
        # add_statistics_step(self.junction_statistics_dict)

        #get_states(self.junction_dict)

        if config['reward_definition'] == 'waiting':
            get_current_reward(self.junction_dict)

        # full_reward = return_reward(self.action_dict, self.junction_dict)
        # self.reward = sum(list(full_reward.values()))
        self.reward = return_reward(self.action_dict, self.junction_dict)
        self.reward_old = self.reward

        full_state = return_states(self.action_dict, self.junction_dict, self.program_dict, self.num_program_dict)
        self.state = [item for sublist in list(full_state.values()) for item in sublist]
        if len(self.state) == 0:
            self.state = self.init_state
        if traci.simulation.getTime() >= config['max_steps']+1000:
            done = True
            traci.close()
        else:
            done = False

        info = {}

        # Return step information
        return self.state, self.reward, done, info

    def _simulate(self, steps_todo):
        for i in range(steps_todo):
            traci.simulationStep()

            # Removed because it slows training down
            # add_statistics_simulate(self.junction_statistics_dict)

            if config['reward_definition'] == 'waiting_fast':
                get_current_reward(self.junction_dict)

            if self.vehicle_stats is not None:
                self.vehicle_stats.update()

            get_states(self.junction_dict)

    def _set_green_phase(self, action_number, tl_id):
        """
        Activate the correct green light combination in sumo
        """
        green_phase_code = action_number * 3  # Obtain the green phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, green_phase_code)

    def _set_yellow_phase(self, old_action, tl_id):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 3 + 1  # Obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, yellow_phase_code)

    def _set_red_phase(self, old_action, tl_id):
        """
        Activate the correct red light combination in sumo
        """
        red_phase_code = old_action * 3 + 2  # Obtain the red phase code, based on the old action
        traci.trafficlight.setPhase(tl_id, red_phase_code)

    def reset(self):
        self._sumo_cmd = configure_sumo(config['gui'], self.model_path, self.model_id, config['sumocfg_file_name'], config['max_steps'])

        try:
            traci.start(self._sumo_cmd)
        except:
            traci.close()
            traci.start(self._sumo_cmd)

        self.sim_length = config['max_steps']
        self.state = [0] * self.num_states
        self.waiting_time, self.queue = return_mean_statistics(self.waiting_time, self.queue, self.junction_statistics_dict)
        return self.state

    def render(self):
        # Implement visualization
        pass

    def get_tl_dicts(self, num_intersections):
        TL_list = {}
        action_dict = {}
        program_dict = {}
        num_program_dict = {}

        for intersection in range(1, num_intersections + 1):
            TL_list[f'TL{intersection}'] = f'{intersection}'
            action_dict[f'TL{intersection}'] = 1
            program_dict[f'TL{intersection}'] = 0
            num_program_dict[f'TL{intersection}'] = 0

        return TL_list, action_dict, program_dict, num_program_dict


def import_sumo_tools():
    """
    Import Python modules from the $SUMO_HOME/tools directory.
    """
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")


def configure_sumo(gui, model_path, model_id, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO.
    """
    # Setting the cmd mode or the visual mode
    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    # Setting the cmd command to run sumo at simulation time
    model_path = os.path.join(f'intersection/{model_path}/model_{model_id}', sumocfg_file_name)
    sumo_cmd = [sumo_binary, "-c", model_path, "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd
