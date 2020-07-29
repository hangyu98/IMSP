"""import modules"""
import csv

import networkx as nx
import numpy as np
import os
from node2vec.node2vec_src import Node2Vec
import pickle
from node2vec.sim_calc import sim_calc
import math


def node2vec_embedding(G, pred, dim, walk_len):
    # --------------------------------------------------------
    # --------------------Configurations----------------------
    # --------------------------------------------------------

    dimension = dim

    walk_length = walk_len

    workers = 8

    # initiate matrix
    list_nodes = []
    for n in G.nodes():
        list_nodes.append(n)

    print('edge weight used: ', pred)
    if pred == 'content':
        # node2vec
        file = open(os.path.abspath(
            '../Heterogeneous Network/data/embeddings/sentence_embedding/sentence_embedding_node.pkl'), 'rb')
        node_emb_dict = pickle.load(file)

        # use Cosine distance to represent node similarity
        score_M = np.empty([len(G.nodes()), len(G.nodes())])
        for src in range(len(G.nodes())):
            for dst in range(len(G.nodes())):
                src_node_vec = node_emb_dict[str(src)]
                dst_node_vec = node_emb_dict[str(dst)]
                # score_M[src][dst] = sim_calc(src_node_vec, dst_node_vec).Cosine()
                score_M[src][dst] = sigmoid(sim_calc(src_node_vec, dst_node_vec).InnerProduct())

        # add weight to graph
        for e in G.edges():
            edge_data = G.get_edge_data(*e)['data']
            if edge_data.__contains__('similarity'):
                G[e[0]][e[1]]['weight'] = float(edge_data['similarity'])
            else:
                G[e[0]][e[1]]['weight'] = float(abs(score_M[list_nodes.index(e[0])][list_nodes.index(e[1])]))

    elif pred == 'structural':
        # node2vec
        node2vec = Node2Vec(G, dimensions=dimension, walk_length=walk_length, workers=workers)
        model = node2vec.fit()

        score_M = np.empty([len(G.nodes()), len(G.nodes())])
        for src in range(len(G.nodes())):
            for dst in range(len(G.nodes())):
                # obtain embeddings word2vec model
                src_node_vec = model.wv.get_vector(str(src))
                dst_node_vec = model.wv.get_vector(str(dst))
                # score_M[src][dst] = sim_calc(src_node_vec, dst_node_vec).Cosine()
                score_M[src][dst] = sigmoid(sim_calc(src_node_vec, dst_node_vec).InnerProduct())

        # add weight to graph
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = float(abs(score_M[list_nodes.index(e[0])][list_nodes.index(e[1])]))

    # -------------------------------
    # embed weight graph
    node2vec_instance = Node2Vec(G, dimensions=dimension, walk_length=walk_length,
                                 workers=workers)

    # store the embedding vector and link score
    with open(os.path.abspath(
            '../Heterogeneous Network/data/embeddings/node2vec_embedding/node2vec_' + pred + '_' + str(dim) + '.csv'),
            'w') as file:
        weight_model = node2vec_instance.fit()
        list_vec = []
        file.write(str(len(G.nodes)) + ' ' + str(dimension) + '\n')
        for n in G.nodes():
            vec = weight_model.wv.get_vector(n)
            vec2lst = list(vec)
            to_append = ''
            for ele in vec2lst:
                to_append = to_append + ' ' + str(ele)
            if len(list_vec) == 0:
                list_vec = np.array(vec)
            else:
                list_vec = np.vstack([list_vec, vec])
            file.write(str(n) + to_append + '\n')


def load_graph(emb_type, dim=128, walk_len=10):
    print('dim: ', dim, 'walk_len: ', walk_len)
    original_G_path = os.path.abspath('../Heterogeneous Network/data/link_prediction/data/original_G.txt')
    G = nx.read_gml(original_G_path)
    node2vec_embedding(G, emb_type, dim, walk_len)
    print('embedding finished\n')


def load_node_embeddings(emb_file_path):
    """
    transform the graph from the file to a dict of {node: node_emb}
    :param emb_file_path:
    :return:
    """
    node_dict = {}
    with open(emb_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        info = next(csv_reader)
        count = int(info[0])
        vec = next(csv_reader)
        for i in range(0, count):
            temp = [float(numeric_string) for numeric_string in vec]
            node_dict[int(temp[0])] = temp[1:]
            if i == count - 1:
                break
            vec = next(csv_reader)

    return node_dict


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    load_graph('content')
    load_graph('structural')
    load_graph('unweighted')
