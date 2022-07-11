"""
File for creating the state representation and reward definition
"""
from xml.dom import minidom

import numpy as np
import traci

from settings import config, StateRepresentation, RewardDefinition


class JunctionManager:
    def __init__(self, traffic_lights: dict):
        self.junction_observers = dict.fromkeys(traffic_lights)
        for junction_name in traffic_lights:
            self.junction_observers[junction_name] = JunctionObserver(
                config['sumocfg_file_name'], junction_name, config['state_representation'],
                'sum', config['reward_definition'], 'sum'
            )

    def receive_states(self) -> None:
        for junction_observer in self.junction_observers.values():
            junction_observer.append_states()

    def receive_rewards(self) -> None:
        for junction_observer in self.junction_observers.values():
            junction_observer.receive_reward()

    def get_state_sizes(self) -> dict:
        state_sizes = dict.fromkeys(self.junction_observers)

        for junction_name, junction_observer in self.junction_observers.items():
            state_sizes[junction_name] = junction_observer.get_state_dimension() + junction_observer.num_programs

        return state_sizes

    def return_states(self) -> dict:
        action_states = dict.fromkeys(self.junction_observers)

        for junction_name, junction_observer in self.junction_observers.items():
            if junction_observer.action == 1:
                aggregated_states = junction_observer.aggregate_states()
                one_hot = np.zeros(junction_observer.num_programs)

                if len(one_hot) > 0:
                    one_hot[junction_observer.program] = 1
                    aggregated_states_one_hot = [*aggregated_states, *one_hot]
                    action_states[junction_name] = aggregated_states_one_hot
                else:
                    aggregated_states.append(traci.trafficlight.getPhase(junction_name))
                    action_states[junction_name] = aggregated_states

                self.junction_observers[junction_name].clear_states()

            elif junction_observer.action == 0:
                action_states[junction_name] = None

        return action_states

    def compute_rewards(self) -> dict:
        action_rewards = dict.fromkeys(self.junction_observers)

        for junction_name, junction_observer in self.junction_observers.items():
            junction_action = junction_observer.action
            if junction_action == 1:
                aggregated_rewards = junction_observer.aggregate_reward()
                action_rewards[junction_name] = aggregated_rewards
                junction_observer.clear_rewards()
            elif junction_action == 0:
                action_rewards[junction_name] = None

        return action_rewards


class JunctionObserver:
    def __init__(
            self, sumo_net, junction_name, state_representation, aggregation_method,
            reward_definition, reward_aggregation
    ):

        self.sumo_net = sumo_net
        self.junction_name = junction_name
        self.state_representation = state_representation
        self.aggregation_method = aggregation_method
        self.reward_definition = reward_definition
        self.reward_aggregation = reward_aggregation

        # Needed for KIVI simulation, but not currently used
        self.action = 1
        self.program = 0
        self.num_programs = 0
        ###

        self.states = []
        self.reward = 0
        self.reward_counter = 0

        self.incoming_lanes = traci.trafficlight.getControlledLanes(junction_name)
        self.incoming_lanes = [lane for lane in self.incoming_lanes if not lane.startswith(':')]
        self.incoming_lanes = list(dict.fromkeys(self.incoming_lanes))
        # self.get_more_lanes(self.incoming_lanes)

    def receive_state(self) -> list:
        state = []

        if self.state_representation == StateRepresentation.VOLUME_LANE_FAST:
            for vehicle_id in traci.simulation.getDepartedIDList():
                traci.vehicle.subscribe(vehicle_id, [traci.constants.VAR_LANE_ID])

            result = traci.vehicle.getAllSubscriptionResults()

            for lane in self.incoming_lanes:
                new_state = sum(result[value][traci.constants.VAR_LANE_ID] == lane for value in result)
                state.append(new_state)

            return state
        elif self.state_representation == StateRepresentation.VOLUME_LANE:
            for lane in self.incoming_lanes:
                state.append(traci.lane.getLastStepVehicleNumber(lane))

        elif self.state_representation == StateRepresentation.WAITING_T:
            for lane in self.incoming_lanes:
                state.append(traci.lane.getWaitingTime(lane))

        elif self.state_representation == StateRepresentation.STAULANGE:
            # Queue length
            pass

        return state

    def get_state_dimension(self) -> int:
        dimension = None
        if self.state_representation in [StateRepresentation.VOLUME_LANE, StateRepresentation.VOLUME_LANE_FAST]:
            dimension = len(self.incoming_lanes)
        elif self.state_representation == StateRepresentation.WAITING_T:
            dimension = len(self.incoming_lanes) + 1

        if self.aggregation_method == "mean":
            dimension = dimension*1

        return dimension

    def aggregate_states(self) -> list:
        states_arr = np.array(self.states)

        if self.aggregation_method == "mean":
            return list(np.mean(states_arr, axis=0))
        elif self.aggregation_method == "sum":
            return list(np.sum(states_arr, axis=0))

    def append_states(self):
        self.states.append(self.receive_state())

    def clear_states(self):
        self.states = []

    def receive_reward(self):
        waiting_total = 0

        if self.reward_definition == RewardDefinition.WAITING:
            for lane in self.incoming_lanes:
                waiting_total += traci.lane.getLastStepHaltingNumber(lane)
        elif self.reward_definition == RewardDefinition.WAITING_FAST:
            for vehicle_id in traci.simulation.getDepartedIDList():
                traci.vehicle.subscribe(vehicle_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])
            result = traci.vehicle.getAllSubscriptionResults()
            for lane in self.incoming_lanes:
                waiting_total += sum(
                    result[value][traci.constants.VAR_LANE_ID] == lane and
                    result[value][traci.constants.VAR_SPEED] < 0.1
                    for value in result
                )

        self.reward -= waiting_total
        self.reward_counter += 1

    def aggregate_reward(self):
        if self.reward_aggregation == "mean":
            return self.reward / self.reward_counter
        elif self.reward_aggregation == "sum":
            return self.reward

    def clear_rewards(self):
        self.reward = 0
        self.reward_counter = 0

    # Function to get further connected lanes from the intersection since first order incLanes may be very short
    def get_more_lanes(self, init_lanes_list):
        add_lanes_list = list()
        for lane_index, init_lane in enumerate(init_lanes_list):
            # Add initial lane to list
            init_edge = traci.lane.getEdgeID(init_lane)
            print(init_lane[-1])
            add_lanes_list.append(init_lane)
            xml_net = minidom.parse(self.sumo_net)
            connection_list = xml_net.getElementsByTagName('connection')
            for s in connection_list:
                print(init_lane[-1])
                if s.attributes['to'].value == init_edge and s.attributes['toLane'] == init_lane[-1]:
                    more_edge = s.attributes['from'].value
                    more_lane_id = s.attributes['fromLane'].value
                    add_lanes_list[lane_index].extend(more_edge+"_"+more_lane_id)
