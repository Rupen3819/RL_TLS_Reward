import sys
import optparse
import random
import traci
import time
import numpy as np
from xml.dom import minidom
from statistics import mean
from utils import *


TL_list = {"TL": 'TL'}
config = import_train_configuration(config_file='training_settings.ini')

def init_statistics(TL_list: dict) -> dict:
    junction_inits = dict.fromkeys(TL_list)
    for junction in TL_list:
        junction_inits[junction]=Statistic_Observer(config['sumocfg_file_name'], junction)
    return junction_inits


def add_statistics_simulate(junction_list: dict) -> None:
    for junction in junction_list:
        junction_list[junction].add_waiting()

    pass

def add_statistics_step(junction_list: dict) -> None:
    for junction in junction_list:
        junction_list[junction].add_queue()
        junction_list[junction].return_waiting()
    pass


def return_mean_statistics(waiting_dict, queue_dict,junction_list) -> [dict, dict]:
    for waiting, queue,junction in zip(waiting_dict, queue_dict,junction_list):
        if waiting_dict[waiting] is None:
            if len(junction_list[junction].waiting_list)!=0:
                waiting_dict[waiting] =list()
                queue_dict[queue] =list()
                waiting_dict[waiting].append(mean(junction_list[junction].waiting_list))
                queue_dict[queue].append(mean(junction_list[junction].queue))
        else:
            if len(junction_list[junction].waiting_list)!=0:
                waiting_dict[waiting].append(mean(junction_list[junction].waiting_list))
                queue_dict[queue].append(mean(junction_list[junction].queue))
        junction_list[junction].queue=list()
        junction_list[junction].waiting_list=list()
    return waiting_dict, queue_dict


class Statistic_Observer():
    def __init__(self,sumo_net, junction_name):
        self.junction_name = junction_name
        self.sumo_net = sumo_net
        self.reward=list()
        self.waiting_time=0
        self.waiting_list=list()
        self.queue=list()
        self.qvalues=list()
        self.sim_step=0

        self.incLanes_list = list(traci.trafficlight.getControlledLanes(junction_name))
        self.incLanes_list = [i for i in self.incLanes_list if i[0] != ":"]
        self.incLanes_list = list(dict.fromkeys(self.incLanes_list))

    def increase_step(self):
        self.sim_step+=1

    def reset_step(self):
        self.sim_step=0

    def add_reward(self, reward):
        pass

    def add_qvalue(self):
        pass

    def add_waiting(self):
        for lane in self.incLanes_list:
            self.waiting_time+=traci.lane.getLastStepHaltingNumber(lane)

    def return_waiting(self):
        self.waiting_list.append(self.waiting_time)
        self.waiting_time=0
        pass

    def add_queue(self):
        length=0
        for lane in self.incLanes_list:
            length+=traci.lane.getLastStepHaltingNumber(lane)
        self.queue.append(length/len(self.incLanes_list))
        pass

    def add_position(self):
        pass
