import math
import os
import random
from xml.dom import minidom

import numpy as np

from utils import Direction
from modular_road_network_structure import is_junction_node


class ModularTrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, network_file):
        self.max_steps = max_steps
        self.n_cars_generated = n_cars_generated
        
        network_xml = minidom.parse(network_file)
        edge_elems = network_xml.getElementsByTagName('edge')
        junction_elems = network_xml.getElementsByTagName('junction')

        self.edges = {
            elem.attributes['id'].value: {
                'from': elem.attributes['from'].value,
                'to': elem.attributes['to'].value,
                'id': elem.attributes['id'].value
            } for elem in edge_elems if not elem.attributes['id'].value.startswith(':')
        }

        self.incoming_edges = {
            edge['to']: edge
            for edge_id, edge in self.edges.items()
        }

        self.outgoing_edges = {
            edge['from']: edge
            for edge_id, edge in self.edges.items()
        }

        self.junction_nodes = {
            elem.attributes['id'].value: {
                'x': float(elem.attributes['x'].value),
                'y': float(elem.attributes['y'].value),
                'id': elem.attributes['id'].value
            } for elem in junction_elems
        }

        self.perimeter_nodes = self.get_perimeter_nodes()

    def get_perimeter_nodes(self) -> dict:
        perimeter_nodes = {
            node_id: dict(node)
            for node_id, node in self.junction_nodes.items()
            if not is_junction_node(node_id)
        }

        nodes = perimeter_nodes.values()
        max_x = max(node['x'] for node in nodes)
        min_x = min(node['x'] for node in nodes)
        max_y = max(node['y'] for node in nodes)
        min_y = min(node['y'] for node in nodes)

        for node in nodes:
            if node['x'] == max_x:
                node['type'] = Direction.EAST
            elif node['x'] == min_x:
                node['type'] = Direction.WEST
            elif node['y'] == max_y:
                node['type'] = Direction.NORTH
            elif node['y'] == min_y:
                node['type'] = Direction.SOUTH

        for node_id, node in perimeter_nodes.items():
            if 'type' not in node:
                incoming_edge = self.incoming_edges[node_id]
                next_junction = incoming_edge['from']
                junction_node = self.junction_nodes[next_junction]
                next_junction_x = junction_node['x']
                next_junction_y = junction_node['y']

                if node['x'] > next_junction_x:
                    node['type'] = Direction.EAST
                elif node['x'] < next_junction_x:
                    node['type'] = Direction.WEST
                elif node['y'] > next_junction_y:
                    node['type'] = Direction.NORTH
                elif node['y'] < next_junction_y:
                    node['type'] = Direction.SOUTH

        return perimeter_nodes

    def generate_routes(self, seed) -> list:
        """
        Generation of the route of every car for one episode.
        """

        direction_to_nodes = {direction: [] for direction in Direction}
        for node in self.perimeter_nodes.values():
            direction_to_nodes[node['type']].append(node)

        """
        We randomly specify if a car comes from the south, north, west or east.
        A car generates has a 30% probability to move straight or move to the left or right end of the system and
        a 10% probability to move to the same direction.
        We only consider transit traffic.
        First we need to create cars according to the Weibull distribution
        """
        np.random.seed(seed)  # Make tests reproducible
        # timings = np.random.weibull(2, self.n_cars_generated)
        timings = np.random.uniform(0, self.max_steps, self.n_cars_generated)
        timings = np.sort(timings)
        car_gen_steps = np.rint(timings)

        """
        We now need to assign the cars to the direction they come from
        """
        routes = []

        directions = [direction for direction in Direction]
        source_probs = np.random.dirichlet(np.ones(4), size=1)[0]

        for car_id in range(self.n_cars_generated):
            source_node, source_direction = self.get_random_node(directions, direction_to_nodes, source_probs)
            source_edge_id = self.outgoing_edges[source_node['id']]['id']

            target_probs = []
            for direction in directions:
                if source_direction == direction:
                    target_probs.append(0.1)
                else:
                    target_probs.append(0.3)

            target_node, _ = self.get_random_node(directions, direction_to_nodes, target_probs)
            target_edge_id = self.incoming_edges[target_node['id']]['id']

            new_car = (car_gen_steps[car_id], source_edge_id, target_edge_id)
            routes.append(new_car)

        return routes

    @staticmethod
    def get_random_node(directions, direction_to_nodes, weights):
        node_direction = random.choices(directions, weights=weights, k=1)[0]

        node = random.choices(
            direction_to_nodes[node_direction],
            weights=np.random.dirichlet(np.ones(len(direction_to_nodes[node_direction])), size=1)[0],
            k=1
        )[0]

        return node, node_direction

    def generate_route_file(self, model_path, seed):
        routes = self.generate_routes(seed)

        xml = minidom.Document()
        routes_elem = xml.createElement('routes')
        xml.appendChild(routes_elem)

        vehicle_type = xml.createElement('vType')
        vehicle_type.setAttribute('accel', '1.0')
        vehicle_type.setAttribute('decel', '4.5')
        vehicle_type.setAttribute('id', 'standard_car')
        vehicle_type.setAttribute('length', '5.0')
        vehicle_type.setAttribute('minGap', '2.5')
        vehicle_type.setAttribute('maxSpeed', '25')
        vehicle_type.setAttribute('sigma', '0.5')
        routes_elem.appendChild(vehicle_type)

        for car_id, route in enumerate(routes):
            car_gen_step, source_edge, target_edge = route
            trip_child = xml.createElement('trip')
            trip_child.setAttribute('id', str(car_id))
            trip_child.setAttribute('depart', str(car_gen_step))
            trip_child.setAttribute('from', source_edge)
            trip_child.setAttribute('to', target_edge)
            trip_child.setAttribute('type', 'standard_car')
            trip_child.setAttribute('departLane', 'random')
            trip_child.setAttribute('departSpeed', '10')

            routes_elem.appendChild(trip_child)

        xml_str = xml.toprettyxml(indent='\t')
        route_file_path = os.path.join(model_path, 'modular_routes.rou.xml')

        with open(route_file_path, 'w') as f:
            f.write(xml_str)

        return route_file_path


# Unused
class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # Number of cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a Weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/dummy/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)


if __name__ == '__main__':
    generator = ModularTrafficGenerator(3600, 1000, 'Modular_Road_Network_Structure/intersection/Ingolstadt_Environment.net.xml')
