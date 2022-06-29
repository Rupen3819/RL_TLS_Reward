import os
import glob
from shutil import copy

from modular_road_network_structure import create_road_network_topology, create_node_xml_file, \
    create_edge_xml_file, create_connection_xml_file, create_tl_logic_xml_file, create_env_xml_file
from settings import config

if __name__ == '__main__':
    num_nodes = 51
    edge_length = 200
    model_path = 'Modular_Road_Network_Structure/intersection'

    files = glob.glob(os.path.join(model_path, '*'))
    for f in files:
        os.remove(f)

    grid_nodes, road_edges, road_connections = create_road_network_topology(num_nodes)

    node_file = create_node_xml_file(model_path, grid_nodes, num_nodes, edge_length)
    edge_file = create_edge_xml_file(model_path, road_edges)
    connection_file = create_connection_xml_file(model_path, road_connections, len(road_edges))
    tl_logic_file = create_tl_logic_xml_file(model_path, grid_nodes)

    create_env_xml_file(model_path, node_file, edge_file, connection_file, tl_logic_file)

    sumo_file = config['sumocfg_file_name']
    copy(f'intersection/{sumo_file}', os.path.join(model_path, sumo_file))

