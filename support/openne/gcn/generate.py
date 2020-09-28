import networkx as nx
import os


def generate_labels():
    G = nx.read_gml(os.path.abspath('../../../data/classifier/original_G.txt'))
    label_dict = {}
    idx = 0
    for node in G.nodes(data=True):
        if not label_dict.keys().__contains__(node[1]['type']):
            label_dict[node[1]['type']] = idx
            print(node[0], idx)
            idx = idx + 1
        else:
            print(node[0], label_dict[node[1]['type']])


if __name__ == '__main__':
    generate_labels()
