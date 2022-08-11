import os
import random
import sys

import traci
from gym import Env
from gym.spaces import Discrete
from sumolib import checkBinary
from scipy.special import softmax

from modular_road_network_structure import create_modular_road_network
from settings import config, RewardDefinition
from state import JunctionManager
from stats.junction import JunctionStatistics, TL_list
from traffic_generation import ModularTrafficGenerator


class SumoEnv(Env):
    def __init__(self, vehicle_stats=None):
        super().__init__()
        self.vehicle_stats = vehicle_stats

        # Create the road topology
        self.model_path, self.model_id = create_modular_road_network(
            config['models_path_name'], config['num_intersections'],
            config['num_actions'], config['intersection_length']
        )

        # Create a generator for creating traffic routes
        self.traffic_generator = ModularTrafficGenerator(
            config['max_steps'],
            config['n_cars_generated'],
            os.path.join(self.model_path, 'environment.net.xml')
        )

        self.traffic_lights = {}
        for i in range(config['num_intersections']):
            self.traffic_lights[get_intersection_name(i)] = str(i + 1)

        self.num_intersections = config['num_intersections']

        self.generate_traffic()
        self._sumo_cmd = configure_sumo(config['gui'], self.model_path, config['sumocfg_file_name'], config['max_steps'])
        traci.start(self._sumo_cmd)

        self.junction_manager = JunctionManager(self.traffic_lights)
        self.junction_stats = JunctionStatistics(self.traffic_lights)

        self.num_states = sum(i + 1 for i in list(self.junction_manager.get_state_sizes().values()))
        self.init_state = [0] * self.num_states
        self.state = [0] * self.num_states

        self.waiting_time = dict.fromkeys(TL_list)
        self.queue = dict.fromkeys(TL_list)

        self.reward = 0

    def step(self, actions):
        # Removed because it slows training down
        # self.junction_stats.step()

        if config['reward_definition'] == RewardDefinition.WAITING:
            self.junction_manager.receive_rewards()

        self.reward = self.junction_manager.compute_rewards()

        full_state = self.junction_manager.return_states()
        self.state = [item for sublist in list(full_state.values()) for item in sublist]

        if len(self.state) == 0:
            self.state = self.init_state

        done = traci.simulation.getTime() >= config['max_steps'] + 1000
        if done:
            traci.close()

        info = {}
        return self.state, self.reward, done, info

    def generate_traffic(self, seed=random.randint(0, 100)):
        self.traffic_generator.generate_route_file(model_path=self.model_path, seed=seed)

    def reset(self):
        self._sumo_cmd = configure_sumo(config['gui'], self.model_path, config['sumocfg_file_name'], config['max_steps'])

        try:
            traci.start(self._sumo_cmd)
        except:
            traci.close()
            traci.start(self._sumo_cmd)

        self.waiting_time, self.queue = self.junction_stats.compute_means(self.waiting_time, self.queue)

        self.state = [0] * self.num_states
        return self.state

    def render(self):
        # Implement visualization
        pass

    def _simulate(self, steps):
        for _ in range(steps):
            traci.simulationStep()

            # Removed because it slows training down
            # self.junction_stats.simulate()

            if config['reward_definition'] == RewardDefinition.WAITING_FAST:
                self.junction_manager.receive_rewards()

            if self.vehicle_stats is not None:
                self.vehicle_stats.update()

            self.junction_manager.receive_states()


class SumoCycleEnv(SumoEnv):
    def __init__(self, vehicle_stats=None):
        super().__init__(vehicle_stats)

        self.green_duration = config['green_duration']
        self.yellow_duration = config['yellow_duration']
        self.red_duration = config['red_duration']

        self.cycle_time = config['cycle_time']

        # num_actions is the same as the num phases in this action space definition
        if config['single_agent']:
            self.single_action_space = config['num_actions']
            self.action_space = config['num_actions'] * len(self.traffic_lights)
            # self.action_space_combinations = list(product(range(0, self.single_action_space.n), repeat=self.num_intersections))
        else:
            self.action_space = config['num_actions']
            self.num_agents = len(self.traffic_lights)

        self.old_actions = dict.fromkeys(self.traffic_lights, [0] * self.action_space)
        self.green_time = self._compute_total_green_time()

    def step(self, raw_actions: list[float]):
        actions = self._compute_cycle_action(raw_actions)

        for light_id in self.traffic_lights:
            self._set_traffic_light_cycle(light_id, actions)

        self._simulate(self.cycle_time)

        return super().step(actions)

    def _set_traffic_light_cycle(self, light_id: str, actions: list[float]):
        light_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(light_id)

        for logic in light_logic:
            print(actions)
            for action in actions:
                for count, phase in enumerate(logic.getPhases()):
                    phase_type = count % 3
                    if phase_type == 0:
                        phase.duration = action + config['min_green_duration']
                    elif phase_type == 1:
                        phase.duration = self.yellow_duration
                    else:
                        phase.duration = self.red_duration

            traci.trafficlight.setCompleteRedYellowGreenDefinition(light_id, logic)

    def _compute_cycle_action(self, raw_action):
        action_proportions = softmax(raw_action)
        action = action_proportions * self.green_time
        return action

    def _compute_total_green_time(self):
        min_phase_duration = config['min_green_duration'] + config['yellow_duration'] + config['red_duration']
        return config['cycle_time'] - self.action_space * min_phase_duration


class SumoPhaseEnv(SumoEnv):
    def __init__(self, vehicle_stats=None):
        super().__init__(vehicle_stats)

        self.green_duration = config['green_duration']
        self.yellow_duration = config['yellow_duration']
        self.red_duration = config['red_duration']

        if config['single_agent']:
            if config['fixed_action_space']:
                self.action_space = Discrete(config['num_actions'])
            else:
                self.single_action_space = Discrete(config['num_actions'])
                self.action_space = Discrete(pow(config['num_actions'], self.num_intersections))
        else:
            self.action_space = Discrete(config['num_actions'])
            self.num_agents = len(self.traffic_lights)

        if config['fixed_action_space'] or not config['single_agent']:
            self._action_generator = self._fixed_action_generator
        else:
            self._action_generator = self._multi_action_generator

        self.old_actions = dict.fromkeys(self.traffic_lights, 0)

    def step(self, actions):
        for (old_action, new_action, traffic_light_id) in self._action_generator(actions):
            if old_action != new_action:
                self._set_yellow_phase(old_action, traffic_light_id)
            else:
                self._set_green_phase(new_action, traffic_light_id)

        self._simulate(self.yellow_duration)

        for (old_action, new_action, traffic_light_id) in self._action_generator(actions):
            if old_action != new_action:
                self._set_red_phase(old_action, traffic_light_id)

        self._simulate(self.red_duration)

        for (old_action, new_action, traffic_light_id) in self._action_generator(actions):
            self._set_green_phase(new_action, traffic_light_id)
            self.old_actions[traffic_light_id] = new_action

        self._simulate(self.green_duration)

        return super().step(actions)

    def _compute_action_combination(self, actions: int):
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

    def _fixed_action_generator(self, actions: dict):
        for light_id in actions:
            yield self.old_actions[light_id], actions[light_id], light_id

    def _multi_action_generator(self, actions: int):
        # action_combination = self.action_space_combinations[actions]
        action_combination = self._compute_action_combination(actions)

        for index, light_id in enumerate(self.traffic_lights):
            yield self.old_actions[light_id], action_combination[index], light_id

    @staticmethod
    def _set_green_phase(old_action, light_id):
        """
        Activate the correct green light combination in SUMO
        """
        green_phase_code = old_action * 3  # Obtain the green phase code, based on the old action
        traci.trafficlight.setPhase(light_id, green_phase_code)

    @staticmethod
    def _set_yellow_phase(old_action, light_id):
        """
        Activate the correct yellow light combination in SUMO
        """
        yellow_phase_code = old_action * 3 + 1  # Obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase(light_id, yellow_phase_code)

    @staticmethod
    def _set_red_phase(old_action, light_id):
        """
        Activate the correct red light combination in SUMO
        """
        red_phase_code = old_action * 3 + 2  # Obtain the red phase code, based on the old action
        traci.trafficlight.setPhase(light_id, red_phase_code)


def get_intersection_name(intersection_index):
    return f'TL{intersection_index + 1}'


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
    sumo_cmd = [
        sumo_binary, "-c", model_path, "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps), "--xml-validation", "never"
    ]

    return sumo_cmd
