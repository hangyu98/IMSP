import networkx as nx
import os as os
from support.word2vec import Text2vec
import pickle


def node_emb():
    original_G_path = os.path.abspath('../../data/prediction/data/original_G.txt')
    full_G = nx.read_gml(original_G_path)
    # returns a dict of {node: attr}
    node_host = nx.get_node_attributes(full_G, 'host')
    node_layer = nx.get_node_attributes(full_G, 'layer')
    node_type = nx.get_node_attributes(full_G, 'type')
    list_attr = []
    for node in full_G.nodes():
        to_add = node_host[node] + ' ' + node_layer[node] + ' ' + node_type[node]
        list_attr.append(to_add)

    # preprocess text2vec model, convert list to list to tokens
    t2v = Text2vec(list_attr)

    # Input: a list of documents, Output: Matrix of vector for all the documents
    # tf-idf = term-frequency inverse document frequency
    docs_emb = t2v.tfidf_weighted_wv()

    print('len: ', len(docs_emb))
    print('dimension: ', len(docs_emb[0]))
    emb_dict = {}
    idx = 0
    for emb in docs_emb:
        emb_dict[str(idx)] = emb
        idx = idx + 1

    with open(os.path.abspath(
            '../../data/embedding_result/sentence_embedding/sentence_embedding_node.pkl'),
              'wb') as file:
        pickle.dump(emb_dict, file)
        file.close()


if __name__ == '__main__':
    node_emb()
