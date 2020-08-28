import networkx as nx
import os as os
from support.word2vec import Text2vec
import pickle


def edge_emb():
    original_G_path = os.path.abspath('../../data/prediction/data/original_G.txt')
    full_G = nx.read_gml(original_G_path)
    # returns a dict of {node: attr}
    node_host = nx.get_node_attributes(full_G, 'host')
    node_layer = nx.get_node_attributes(full_G, 'layer')
    node_type = nx.get_node_attributes(full_G, 'type')
    edge_list_attr = []
    # construct sentences for pairs
    print('now constructing sentences')
    for node in full_G.nodes():
        for node_2 in full_G.nodes():
            if node != node_2:
                basic_info = node_host[node] + ' ' + node_layer[node] + ' ' + node_type[node] + ' ' + node_host[
                    node_2] + ' ' + node_layer[node_2] + ' ' + node_type[node_2]
                if (str(node), str(node_2)) in full_G.edges():
                    if full_G[node][node_2]['relation'].__contains__('similar'):
                        to_add = basic_info + ' homogeneous similarity'
                    else:
                        to_add = basic_info + ' heterogeneous ' + \
                                 full_G[node][node_2]['relation']
                else:
                    if node_type[node] == node_type[node_2]:
                        to_add = basic_info + ' homogeneous similarity'
                    else:
                        if (node_layer[node] == 'virus' and node_layer[node_2] == 'host') or (
                                node_layer[node] == 'host' and node_layer[node_2] == 'virus'):
                            to_add = basic_info + ' heterogeneous infects'
                        elif (node_layer[node] == 'virus' and node_layer[node_2] == 'virus protein') or (
                                node_layer[node] == 'virus protein' and node_layer[node_2] == 'virus'):
                            to_add = basic_info + ' heterogeneous belongs'
                        elif (node_layer[node] == 'host' and node_layer[node_2] == 'host protein') or (
                                node_layer[node] == 'host protein' and node_layer[node_2] == 'host'):
                            to_add = basic_info + ' heterogeneous belongs'
                        elif (node_layer[node] == 'host protein' and node_layer[node_2] == 'virus protein') or (
                                node_layer[node] == 'virus protein' and node_layer[node_2] == 'host protein'):
                            to_add = node_host[node] + ' ' + node_layer[node] + ' ' + node_type[node] + ' ' + node_host[
                                node_2] + ' ' + node_layer[node_2] + ' ' + node_type[
                                         node_2] + ' heterogeneous interacts'
                        else:
                            to_add = basic_info + ' unconnected'
                edge_list_attr.append(to_add)
                print('to_add: ', to_add)

    # extract index to node pair dictionary
    index2pair_dict = {}
    src_node = 0
    dst_node = 0
    count = 0
    num_of_nodes = len(full_G.nodes())

    while count < len(edge_list_attr):
        # remove loops
        if src_node == dst_node:
            dst_node = (dst_node + 1) % num_of_nodes
            continue
        # add new node pair to the dict
        index2pair_dict[count] = (src_node, dst_node)
        # inc
        dst_node = dst_node + 1
        if dst_node == num_of_nodes:
            src_node = src_node + 1
            dst_node = 0
        count = count + 1

    print('Now training Text2vec model')
    # preprocess text2vec model, convert list to list to tokens
    t2v = Text2vec(edge_list_attr)
    print('Training Text2vec model finished')

    # Input: a list of documents, Output: Matrix of vector for all the documents
    # tf-idf = term-frequency inverse document frequency
    docs_emb = t2v.tfidf_weighted_wv()

    # store embedding results to a dict
    edge_emb_dict = {}
    for n in range(len(docs_emb)):
        node_pair = index2pair_dict[n]
        edge_emb_dict[node_pair] = docs_emb[n]

    print('Now saving result to file')
    # save the dict to disk
    with open(os.path.abspath('../../data/embedding_result/sentence_embedding/sentence_embedding.pkl'),
              'wb') as file:
        pickle.dump(edge_emb_dict, file)
        file.close()
    print("Result saved")


if __name__ == '__main__':
    edge_emb()
