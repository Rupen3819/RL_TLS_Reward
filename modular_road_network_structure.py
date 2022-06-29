import math
import os
from shutil import copy
from xml.dom import minidom

from settings import config
from utils import create_intersection_path, Direction, RelativeDirection


def is_junction_node(node_name):
    return not node_name.startswith('c')


def add_bidirectional_edges(edges, junction_out_roads, src_node, dest_node, direction):
    if is_junction_node(src_node) or is_junction_node(dest_node):
        prefix = 'ce'
    else:
        prefix = 'e'

    out_edge = f'{prefix}{len(edges) + 1}'
    edges[out_edge] = (src_node, dest_node)

    in_edge = f'{prefix}{len(edges) + 1}'
    edges[in_edge] = (dest_node, src_node)

    if is_junction_node(src_node):
        if src_node not in junction_out_roads:
            junction_out_roads[src_node] = {}
        junction_out_roads[src_node][direction] = out_edge

    if is_junction_node(dest_node):
        if dest_node not in junction_out_roads:
            junction_out_roads[dest_node] = {}
        junction_out_roads[dest_node][direction.opposite()] = in_edge

    return out_edge, in_edge


def create_road_network_topology(num_nodes):
    # Map from traffic light name (i.e. '3') to its grid coordinates
    junction_nodes: dict[str, (int, int)] = {}

    # Map from name of unregulated node (i.e 'c8') to its grid coordinates
    perimeter_nodes: dict[str, (int, int)] = {}

    # Reverse map of junction_nodes (maps coordinates to traffic light name)
    node_from_coordinates: dict[(int, int), str] = {}

    # Map from traffic light name, to a map from direction (i.e. NORTH) to the road edge name
    # of the road travelling in that direction away from the junction
    junction_out_roads: dict[str, dict:[Direction, str]] = {}

    # Map from road edge name to a 2-tuple
    road_edges: dict[str, (str, str)] = {}

    max_y = math.ceil(math.sqrt(num_nodes))  # The max y coordinate of any node in the network
    max_x = round(math.sqrt(num_nodes))  # The max x coordinate of any node in the network

    corner_index = 0  # Index of the top-right square in the grid for the current layer
    corner_increment = 2  # Used to easily compute the next value of corner_index

    # The current layer of the square grid. Layers are added first from left to right on the top row,
    # and then top to bottom on the right column
    layer = 1

    for i in range(num_nodes):
        node_name = str(i + 1)

        if i < corner_index:
            # Current square is on the top row of the grid
            x, y = i - (layer - 1) ** 2 + 1, layer
            dest_node_name = node_from_coordinates[x, y - 1]
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, dest_node_name, Direction.SOUTH)
        elif i > corner_index:
            # Current square is on the rightmost column of the grid
            x, y = layer, layer ** 2 - i
            dest_node_name = node_from_coordinates[x, y + 1]
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, dest_node_name, Direction.NORTH)
        else:  # i == corner_index
            # Current square is the top right square of the grid
            x, y = layer, layer

        if x == 1:
            # Current square is in the leftmost column of the grid
            extra_node_name = f'c{num_nodes + len(perimeter_nodes) + 1}'
            perimeter_nodes[extra_node_name] = (x - 0.75, y)
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, extra_node_name, Direction.WEST)
        elif x > 1:
            # Current square is in any other column of the grid
            dest_node_name = node_from_coordinates[x - 1, y]
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, dest_node_name, Direction.WEST)

        if (
            x == max_x or
            (i >= corner_index and i + corner_increment + 1 >= num_nodes) or
            (i == num_nodes - 1 and i <= corner_index)
        ):
            # Current square is in the rightmost column of the grid (which may not be a full column)
            # OR current square is in the current column, and no square will later be placed to the right of it
            # OR current square is the last square, and is in the top column
            extra_node_name = f'c{num_nodes + len(perimeter_nodes) + 1}'
            perimeter_nodes[extra_node_name] = (x + 0.75, y)
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, extra_node_name, Direction.EAST)

        if (
            y == max_y or
            (i <= corner_index and i + corner_increment - 1 >= num_nodes)
        ):
            # Current square is in the top row (which may not be a full row)
            # OR current square is in the current row, and no square will later be placed above it
            extra_node_name = f'c{num_nodes + len(perimeter_nodes) + 1}'
            perimeter_nodes[extra_node_name] = (x, y + 0.75)
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, extra_node_name, Direction.NORTH)

        if (
            y == 1 or
            (i == num_nodes - 1 and i >= corner_index)
        ):
            # Current square is in the bottom row of the grid
            # OR current square is the last square, and is in the rightmost column
            extra_node_name = f'c{num_nodes + len(perimeter_nodes) + 1}'
            perimeter_nodes[extra_node_name] = (x, y - 0.75)
            add_bidirectional_edges(road_edges, junction_out_roads, node_name, extra_node_name, Direction.SOUTH)

            # This is the last square in the layer, so we increase layer, corner_index, and corner_increment
            layer += 1
            corner_index += corner_increment
            corner_increment += 2

        junction_nodes[node_name] = (x, y)
        node_from_coordinates[x, y] = node_name

    road_connections = []

    for edge_name, (from_node, to_node) in road_edges.items():
        if is_junction_node(to_node):
            next_edges = []

            in_direction = None
            for (direction, next_edge_name) in junction_out_roads[to_node].items():
                next_edge_src, next_edge_dest = road_edges[next_edge_name]
                if from_node == next_edge_dest:
                    in_direction = direction
                else:
                    next_edges.append((direction, next_edge_name))

            if in_direction is None:
                continue

            for out_direction, next_edge_name in next_edges:
                relative_direction = out_direction.relative(in_direction)
                connection = (edge_name, next_edge_name, relative_direction)
                road_connections.append(connection)

    return junction_nodes | perimeter_nodes, road_edges, road_connections


def create_node_xml_file(model_path, grid_nodes, num_nodes, edge_length):
    xml = minidom.Document()
    nodes = xml.createElement('nodes')
    xml.appendChild(nodes)

    """Create nod.xml file"""
    for node_element, (x, y) in grid_nodes.items():
        node_x = x * edge_length
        node_y = y * edge_length

        node_child = xml.createElement('node')
        node_child.setAttribute('id', node_element)
        node_child.setAttribute('x', f'{node_x}')
        node_child.setAttribute('y', f'{node_y}')
        node_child.setAttribute('type', 'traffic_light' if is_junction_node(node_element) else 'unregulated')
        if is_junction_node(node_element):
            node_child.setAttribute('tl', f'TL{node_element}')

        nodes.appendChild(node_child)

    xml_str = xml.toprettyxml(indent="\t")
    node_file_path = os.path.join(model_path, f'Ingolstadt_{num_nodes}_Nodes.nod.xml')

    with open(node_file_path, "w") as f:
        f.write(xml_str)

    return node_file_path


def create_edge_xml_file(model_path, road_edges):
    xml = minidom.Document()
    edges = xml.createElement('edges')
    xml.appendChild(edges)

    """Create edg.xml file"""
    for edge_id, (edge_from, edge_to) in road_edges.items():
        # if edge_id in ['ce45']: continue

        edge_child = xml.createElement('edge')
        edge_child.setAttribute('id', edge_id)
        edge_child.setAttribute('from', edge_from)
        edge_child.setAttribute('to', edge_to)
        edge_child.setAttribute('numLanes', '4')
        edge_child.setAttribute('speed', '13.9')

        edges.appendChild(edge_child)

    xml_str = xml.toprettyxml(indent="\t")
    edge_file_path = os.path.join(model_path, f'Ingolstadt_{len(road_edges) // 2}_Edges.edg.xml')

    with open(edge_file_path, "w") as f:
        f.write(xml_str)

    return edge_file_path


def create_connection_xml_file(model_path, road_connections, num_edges):
    xml = minidom.Document()
    connections = xml.createElement('connections')
    xml.appendChild(connections)

    def create_connection_child(edge_from, edge_to, lane_from, lane_to):
        nonlocal connections

        connection_child = xml.createElement('connection')
        connection_child.setAttribute('from', edge_from)
        connection_child.setAttribute('to', edge_to)
        connection_child.setAttribute('fromLane', lane_from)
        connection_child.setAttribute('toLane', lane_to)
        connections.appendChild(connection_child)

    """Define lane specific connections for edges"""
    for from_edge, to_edge, relative_direction in road_connections:
        if relative_direction == RelativeDirection.STRAIGHT:
            for lane in range(0, 3):
                create_connection_child(edge_from=from_edge, edge_to=to_edge, lane_from=f'{lane}', lane_to=f'{lane}')
        else:
            lane_from_to = '0' if relative_direction == RelativeDirection.RIGHT else '3'
            create_connection_child(edge_from=from_edge, edge_to=to_edge, lane_from=lane_from_to, lane_to=lane_from_to)

    xml_str = xml.toprettyxml(indent="\t")
    connection_file_path = os.path.join(model_path, f'Ingolstadt_{num_edges}_Connections.con.xml')

    with open(connection_file_path, "w") as f:
        f.write(xml_str)

    return connection_file_path


def create_tl_logic_xml_file(model_path, grid_nodes):
    xml = minidom.Document()
    tl_logics = xml.createElement('tlLogics')
    xml.appendChild(tl_logics)

    junction_nodes = [node for node in grid_nodes.keys() if is_junction_node(node)]

    phase_xml_states = [
        'GGGGrrrrrrGGGGrrrrrr', 'yyyyrrrrrryyyyrrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrGrrrrrrrrrGrrrrr', 'rrrryrrrrrrrrryrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrGGGGrrrrrrGGGGr', 'rrrrryyyyrrrrrryyyyr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrrrrrGrrrrrrrrrG', 'rrrrrrrrryrrrrrrrrry', 'rrrrrrrrrrrrrrrrrrrr',
        'GGGGGrrrrrrrrrrrrrrr', 'yyyyyrrrrrrrrrrrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrGGGGGrrrrrrrrrr', 'rrrrryyyyyrrrrrrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrrrrrrGGGGGrrrrr', 'rrrrrrrrrryyyyyrrrrr', 'rrrrrrrrrrrrrrrrrrrr',
        'rrrrrrrrrrrrrrrGGGGG', 'rrrrrrrrrrrrrrryyyyy', 'rrrrrrrrrrrrrrrrrrrr',
    ]

    """Create tll.xml file"""
    for node in junction_nodes:
        tl_logic_id = f'TL{node}'

        tl_logic_child = xml.createElement('tlLogic')
        tl_logic_child.setAttribute('id', tl_logic_id)
        tl_logic_child.setAttribute('type', 'static')
        tl_logic_child.setAttribute('programID', '0')
        tl_logic_child.setAttribute('offset', '0')

        for phase_element in phase_xml_states:
            phase_child = xml.createElement('phase')
            phase_child.setAttribute('duration', '100')
            phase_child.setAttribute('state', phase_element)
            tl_logic_child.appendChild(phase_child)

        tl_logics.appendChild(tl_logic_child)

    xml_str = xml.toprettyxml(indent="\t")

    tl_logic_file_path = os.path.join(model_path, f'Ingolstadt_{len(junction_nodes)}_TrafficLights.tll.xml')

    with open(tl_logic_file_path, "w") as f:
        f.write(xml_str)

    return tl_logic_file_path


def create_env_xml_file(model_path, node_file, edge_file, connection_file, tl_logic_file):
    env_file_path = os.path.join(model_path, 'environment.net.xml')
    os.system(
        f'netconvert --node-files={node_file} --edge-files={edge_file} --connection-files={connection_file} '
        f'--tllogic-files={tl_logic_file} -o {env_file_path}'
    )
    return env_file_path


def create_modular_road_network(models_path_name, num_nodes, edge_length=100):
    model_path, model_id = create_intersection_path(models_path_name)

    grid_nodes, road_edges, road_connections = create_road_network_topology(num_nodes)

    node_file = create_node_xml_file(model_path, grid_nodes, num_nodes, edge_length)
    edge_file = create_edge_xml_file(model_path, road_edges)
    connection_file = create_connection_xml_file(model_path, road_connections, len(road_edges))
    tl_logic_file = create_tl_logic_xml_file(model_path, grid_nodes)

    create_env_xml_file(model_path, node_file, edge_file, connection_file, tl_logic_file)

    sumo_file = config['sumocfg_file_name']
    copy(f'intersection/{sumo_file}', os.path.join(model_path, sumo_file))

    return model_path, model_id


if __name__ == '__main__':
    create_modular_road_network('models', 50, 200)
