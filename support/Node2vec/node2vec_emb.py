"""import modules"""
import copy
import csv

import networkx as nx
import numpy as np
import os
import pickle

import support.Node2vec.utils

from support.Node2vec.node2vec import Node2Vec
from support.Node2vec.utils import sigmoid


def node2vec_embedding(G, weighted, dim, walk_len, num_walks, p, q):
    # --------------------------------------------------------
    # --------------------Configurations----------------------
    # --------------------------------------------------------

    dimension = dim

    walk_length = walk_len

    workers = 8

    num_walks = num_walks

    preprocess(G, weighted)

    if weighted == 'unweighted':
        node2vec_instance = Node2Vec(G, dimensions=dimension, num_walks=num_walks, walk_length=walk_length,
                                     workers=workers, quiet=True)
        # save the embedding vector and link score
        with open(os.path.abspath(
                'data/embedding/Node2vec.csv'), 'w') as file:
            save_res(G, dimension, file, node2vec_instance, walk_len)
    else:
        node2vec_instance = Node2Vec(G, dimensions=dimension, num_walks=num_walks, walk_length=walk_length,
                                     workers=workers, p=p, q=q, quiet=True)
        # save the embedding vector and link score
        with open(os.path.abspath(
                'data/embedding/CrossNELP.csv'), 'w') as file:
            save_res(G, dimension, file, node2vec_instance, walk_len)


def save_res(G, dimension, file, node2vec_instance, walk_len):
    weight_model = node2vec_instance.fit(window_size=walk_len)
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


def preprocess(G, weighted):
    print("weighted? ", weighted == 'weighted')
    # initiate matrix
    list_nodes = []
    for n in G.nodes():
        list_nodes.append(n)
    if weighted == 'weighted':
        print('assigning edge weights...')
        # Node2vec
        file = open(os.path.abspath(
            'data/embedding/sentence_embedding/sentence_embedding_node.pkl'), 'rb')
        node_emb_dict = pickle.load(file)

        value_lst = []
        for src in range(len(G.nodes())):
            for dst in range(len(G.nodes())):
                src_node_vec = node_emb_dict[str(src)]
                dst_node_vec = node_emb_dict[str(dst)]
                value_lst.append(support.Node2vec.utils.sim_calc(src_node_vec, dst_node_vec).TS_SS())

        mean = np.mean(value_lst)
        print('mean: ', mean)
        # use Cosine distance to represent node similarity
        score_M = np.empty([len(G.nodes()), len(G.nodes())])
        for src in range(len(G.nodes())):
            for dst in range(len(G.nodes())):
                src_node_vec = node_emb_dict[str(src)]
                dst_node_vec = node_emb_dict[str(dst)]
                score_M[src][dst] = sigmoid(
                    support.Node2vec.utils.sim_calc(src_node_vec, dst_node_vec).TS_SS() / mean)

        # add weight to network
        for e in G.edges():
            edge_relation = G.get_edge_data(*e)['relation']
            if edge_relation.__contains__('similar'):
                G[e[0]][e[1]]['weight'] = float(G.get_edge_data(*e)['similarity']) / 100
            else:
                G[e[0]][e[1]]['weight'] = float(abs(score_M[list_nodes.index(e[0])][list_nodes.index(e[1])]))
    else:
        print("no need to assign edge weight")

    nx.write_adjlist(G, os.path.abspath('data/embedding/adjlist.txt'))


def load_node_embeddings(emb_file_path):
    """
    transform the network from the file to a dict of {node: node_emb}
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


def node_structure_emb(weighted, dim=128, walk_len=6, num_walks=50, p=1, q=1):
    original_G_path = os.path.abspath('data/classifier/original_G.txt')
    print('weighted? ', weighted, 'dim: ', dim, '\twalk_len: ', walk_len, '\tp: ', p, '\tq: ', q, '\tnum_walks',
          num_walks)
    G = nx.read_gml(original_G_path)

    node2vec_embedding(G, weighted, dim, walk_len, num_walks, p, q)
    print('Node2vec ' + weighted + ' embedding finished')

# if __name__ == '__main__':
#     node_structure_emb('weighted', dim=128, walk_len=5, num_walks=30)
#     node_structure_emb('unweighted', dim=128, walk_len=5, num_walks=30)
