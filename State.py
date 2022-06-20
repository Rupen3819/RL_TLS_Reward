"""
File for creating the state representation and reward definition
"""
import sys
import optparse
import random
import traci
import time
import numpy as np
from xml.dom import minidom
from utils import *

config = import_configuration()
TL_list = {"3030": 'cluster_25579770_2633530003_2633530004_2633530005'}


def init_states(TL_list: dict) -> dict:
    # Initialze dict with available juctions
    junction_inits = dict.fromkeys(TL_list)
    for junction in TL_list:
        junction_inits[junction]=State_Observer(config['sumocfg_file_name'], junction, config['state_representation'], 'sum',
                                          config['reward_definition'], 'sum')
    return junction_inits


def get_states(junction_list: dict) -> None:
    for junction in junction_list:
        junction_list[junction].append_states()


def get_current_reward(junction_list: dict) -> None:
    for junction in junction_list:
        junction_list[junction].get_rewards()


def get_state_size(junction_list: dict,  num_programs: dict) -> dict:
    state_size_dict = dict.fromkeys(junction_list)
    for junction in junction_list:
        state_size_dict[junction]=junction_list[junction].get_state_dimension() + num_programs[junction]
    return state_size_dict


def return_states(action_dict: dict, junction_list: dict, program_dict: dict, num_programs: dict) -> dict:
    action_states_dict = dict.fromkeys(junction_list)
    for junction in junction_list:
        if action_dict[junction] == 1:
            aggregated_states = junction_list[junction].aggregate_states()
            one_hot=np.zeros(num_programs[junction])
            if len(one_hot) > 0:
                one_hot[program_dict[junction]] = 1
                aggregated_states_one_hot = [*aggregated_states, *one_hot]
                action_states_dict[junction] = aggregated_states_one_hot
            else:
                aggregated_states.append(traci.trafficlight.getPhase(junction))
                action_states_dict[junction] = aggregated_states
            junction_list[junction].clear_states()
        if action_dict[junction] == 0:
            action_states_dict[junction] = None
    return action_states_dict

def return_reward(action_dict: dict, junction_list: dict) -> dict:
    action_reward_dict=dict.fromkeys(junction_list)
    for junction in junction_list:
        if action_dict[junction] == 1:
            aggregated_rewards=junction_list[junction].aggregate_reward()
            action_reward_dict[junction]=aggregated_rewards
            junction_list[junction].clear_rewards()
        if action_dict[junction]==0:
            action_reward_dict[junction] = None
    return action_reward_dict


#Function to add a one-hot vector for the programs of the active junctions
def onehot_program(program_list: dict,num_programs: int):
    programs = np.array(program_list)
    programs_ = np.zeros((programs.size, num_programs))
    programs_[np.arange(programs.size),programs] = 1
    return programs_.tolist()


class State_Observer():
    def __init__(self, sumo_net, junction_name, state_representation, aggregation_method,
                 reward_definition, reward_aggregation):

        self.junction_name = junction_name
        self.state_representation = state_representation
        self.sumo_net = sumo_net
        self.aggregation_method = aggregation_method
        self.states = list()

        self.reward=0
        self.reward_counter=0
        self.reward_definition=reward_definition
        self.reward_aggregation=reward_aggregation

        self.incLanes_list = list(traci.trafficlight.getControlledLanes(junction_name))
        self.incLanes_list = [i for i in self.incLanes_list if i[0]!=":"]
        self.incLanes_list = list(dict.fromkeys(self.incLanes_list))
#        self.get_more_lanes(self.incLanes_list)


##        xml_net=minidom.parse(self.sumo_net)
##        junction_list=xml_net.getElementsByTagName('junction')
##        for s in junction_list:
##            if s.attributes['id'].value==self.junction_name:
##                incLanes=s.attributes['incLanes'].value
##        tabs=[m.start()+1 for m in re.finditer(' ', incLanes)]
##        tabs.insert(0,0)
##        self.incLanes_list=list()
##        for tab in range(len(tabs)):
##            if incLanes[tabs[tab]]!=":":
##                try:
##                    self.incLanes_list.append(incLanes[tabs[tab]:tabs[tab+1]-1])
##                except:
##                    self.incLanes_list.append(incLanes[tabs[tab]:len(incLanes)])


    def get_incLanes(self):
        print(self.incLanes_list)


    def get_state(self) -> list:
        state=list()
        #print(traci.vehicle.getSubscriptionResults("0xd3"))
        if self.state_representation == "volume_lane_fast":
            for veh_id in traci.simulation.getDepartedIDList():
                traci.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID])
            result = traci.vehicle.getAllSubscriptionResults()

            for lane in self.incLanes_list:
                new=sum(result[value][81] == lane for value in result)
                state.append(new)

            return state
        if self.state_representation == "volume_lane":
            for lane in self.incLanes_list:
                state.append(traci.lane.getLastStepVehicleNumber(lane))

        if self.state_representation == "waiting_t":
            for lane in self.incLanes_list:
                state.append(traci.lane.getWaitingTime(lane))

        if self.state_representation == "Staulänge":
            pass
        return state

    def get_state_dimension(self) -> int:
        dimension=None
        if self.state_representation == "volume_lane":
            dimension = len(self.incLanes_list)
        if self.state_representation == "waiting_t":
            dimension = len(self.incLanes_list) + 1
        if self.state_representation == "volume_lane_fast":
            dimension = len(self.incLanes_list)

        if self.aggregation_method == "mean":
            dimension = dimension*1

        return dimension

    def aggregate_states(self) -> list:
        if self.aggregation_method == "mean":
            states_arr=np.array(self.states)
            agg_states_arr=np.mean(states_arr, axis=0)
            return list(agg_states_arr)

        if self.aggregation_method == "sum":
            states_arr = np.array(self.states)
            agg_states_arr = np.sum(states_arr, axis=0)
            return list(agg_states_arr)

    def append_states(self):
        self.states.append(self.get_state())

    def clear_states(self):
        self.states=list()

    def get_rewards(self):
        self.reward_counter+=1
        if self.reward_definition=="waiting":
            waiting_total=0
            for lane in self.incLanes_list:
                waiting_total += traci.lane.getLastStepHaltingNumber(lane)
        if self.reward_definition == "waiting_fast":
            waiting_total = 0
            for veh_id in traci.simulation.getDepartedIDList():
                traci.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])
            result = traci.vehicle.getAllSubscriptionResults()
            #print(len(result))
            for lane in self.incLanes_list:
                waiting_total+=sum(result[value][81] == lane and result[value][64] < 0.1 for value in result)
        self.reward += -waiting_total

    def aggregate_reward(self):
        if self.reward_aggregation=="mean":
            agg_reward=self.reward/self.reward_counter
            return agg_reward

        if self.reward_aggregation=="sum":
            agg_reward=self.reward
            return agg_reward


    def clear_rewards(self):
        self.reward=0
        self.reward_counter=0


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
#
#
#
#
#
# # we need to import python modules from the $SUMO_HOME/tools directory
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")
#
# from sumolib import checkBinary  # noqa
# import traci  # noqa
#
#
#
# def run(selected_junctions):
#     """execute the TraCI control loop"""
#     step = 0
#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()
#         get_states(selected_junctions)
#         get_current_reward(selected_junctions)
#         #print(len(selected_junctions["3030"].states))
#         step += 1
#         if step % 120 == 0:
#             print(len(selected_junctions["3030"].states))
#             state_return=return_states({"3030": 1}, selected_junctions, {"3030": 3}, {"3030": 5})
#             reward_return=return_reward({"3030": 1}, selected_junctions)
#             print(state_return)
#             print(reward_return)
#             print(len(selected_junctions["3030"].states))
#     traci.close()
#     sys.stdout.flush()
#
#
# def get_options():
#     optParser = optparse.OptionParser()
#     optParser.add_option("--nogui", action="store_true",
#                          default=False, help="run the commandline version of sumo")
#     options, args = optParser.parse_args()
#     return options
#
#
# # this is the main entry point of this script
# if __name__ == "__main__":
#
#     options = get_options()
#
#     # this script has been called from the command line. It will start sumo as a
#     # server, then connect and run
#     #if False:
#     #sumoBinary = checkBinary('sumo')
#     #else:
#     sumoBinary = checkBinary('sumo-gui')
#
#     # this is the normal way of using traci. sumo is started as a
#     # subprocess and then the python script connects and runs
#     traci.start([sumoBinary, "-c", "SUMO/config_24h_vehicle.sumocfg",
#                              "--tripinfo-output", "tripinfo.xml"])
#     selected_junctions = init_states(TL_list)#(['3040',"31065768"])
#     state_size = get_state_size(selected_junctions, {"3030": 5})
#     run(selected_junctions)

