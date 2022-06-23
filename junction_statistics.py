import traci
from statistics import mean

TL_list = {"TL": 'TL'}


class JunctionStatistics:
    def __init__(self, traffic_lights: dict):
        self.junction_observers = dict.fromkeys(traffic_lights)
        for junction_name in traffic_lights:
            self.junction_observers[junction_name] = JunctionObserver(junction_name)

    def simulate(self) -> None:
        for junction_observer in self.junction_observers.values():
            junction_observer.add_waiting()

    def step(self) -> None:
        for junction_observer in self.junction_observers.values():
            junction_observer.add_queue()
            junction_observer.return_waiting()

    def compute_means(self, waiting_dict, queue_dict) -> [dict, dict]:
        for waiting, queue, junction in zip(waiting_dict, queue_dict, self.junction_observers):
            if waiting_dict[waiting] is None:
                if len(self.junction_observers[junction].waiting_list) != 0:
                    waiting_dict[waiting] = []
                    queue_dict[queue] = []
                    waiting_dict[waiting].append(mean(self.junction_observers[junction].waiting_list))
                    queue_dict[queue].append(mean(self.junction_observers[junction].queue))
            else:
                if len(self.junction_observers[junction].waiting_list) != 0:
                    waiting_dict[waiting].append(mean(self.junction_observers[junction].waiting_list))
                    queue_dict[queue].append(mean(self.junction_observers[junction].queue))

            self.junction_observers[junction].queue = []
            self.junction_observers[junction].waiting_list = []

        return waiting_dict, queue_dict
            

class JunctionObserver:
    def __init__(self, junction_name):
        self.junction_name = junction_name

        self.waiting_time = 0
        self.waiting_list = []
        self.queue = []

        self.incoming_lanes = traci.trafficlight.getControlledLanes(junction_name)
        self.incoming_lanes = [lane for lane in self.incoming_lanes if not lane.startswith(':')]
        self.incoming_lanes = list(dict.fromkeys(self.incoming_lanes))

    def add_waiting(self):
        for lane in self.incoming_lanes:
            self.waiting_time += traci.lane.getLastStepHaltingNumber(lane)

    def return_waiting(self):
        self.waiting_list.append(self.waiting_time)
        self.waiting_time = 0

    def add_queue(self):
        length = 0
        for lane in self.incoming_lanes:
            length += traci.lane.getLastStepHaltingNumber(lane)
        self.queue.append(length / len(self.incoming_lanes))
