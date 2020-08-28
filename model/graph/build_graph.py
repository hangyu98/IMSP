import json
import os as os
from graph import build_graph_utils as bd
import networkx as nx

# --------------------------------------------------------
# --------------------Configurations----------------------
# --------------------------------------------------------


# file directory for homo similarity matrices
file_dir = os.path.abspath('../../data/similarity_matrix')


def build_graph(original_G_path, list_of_hosts, list_of_viruses):
    # --------------------------------------------------------
    # --------------------Build the graph---------------------
    # --------------------------------------------------------
    print("----------------------------------------")
    print("BUILDING GRAPH")
    print("----------------------------------------")

    # Initialize directed graph
    G = nx.Graph()

    # initialize a dict {virus: virus proteins}
    dict_of_belong_relations = {host: [] for host in list_of_hosts}

    dict_of_belong_relations.update({host: [] for host in list_of_viruses})

    # initialize list of nodes groups and edge groups
    dict_of_nodes_groups, dict_of_edges_groups = {'virus': {}, 'host protein': {}, 'virus protein': {}, 'host': {}}, \
                                                 {'virus': {}, 'host protein': {}, 'virus protein': {}, 'host': {}}

    # fill in these two nested dictionaries by building homo graphs
    bd.build_home_graph(G, dict_of_nodes_groups=dict_of_nodes_groups, dict_of_edges_groups=dict_of_edges_groups,
                        file_dir=file_dir, belong_relation_dict=dict_of_belong_relations)

    # heterogeneous edges
    hetero_edges = []
    bd.build_original_hetero_edges(G, hetero_edges, dict_of_belong_relations, dict_of_nodes_groups, list_of_viruses)

    # convert to and store cytoscape needed format
    # print("Saving to cytoscape required json...\n")
    json_cyto = nx.cytoscape_data(G)
    with open('../../data/cytoscape/cytoscape_undirected_original.json', 'w') as json_file:
        json.dump(json_cyto, json_file)

    nx.write_gml(G, original_G_path)
    for node in G:
        to_print = ''
        for ele in list(nx.neighbors(G, node)):
            to_print = to_print + ' ' + ele
        print(node, to_print)
    print('original graph saved!')
    print("graph building finished!")
