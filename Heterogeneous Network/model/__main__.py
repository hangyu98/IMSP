"""import modules"""

import os as os
from graph import build_graph

from utils import process, model_eval, model_pred


def main():
    # --------------------------------------------------------
    # --------------------Configurations----------------------
    # --------------------------------------------------------

    # paths for structural embeddings
    structural_emb_path = [
        os.path.abspath('../data/embeddings/deepwalk_embedding/deepwalk_unweighted_128.csv'),
        os.path.abspath('../data/embeddings/node2vec_embedding/node2vec_weighted_structural_128.csv'),
        os.path.abspath('../data/embeddings/node2vec_embedding/node2vec_weighted_content_128.csv'),
        os.path.abspath('../data/embeddings/node2vec_embedding/node2vec_unweighted_128.csv'),
        os.path.abspath('../data/embeddings/line_embedding/line_unweighted_128.csv'),
    ]

    # list of hosts
    list_of_hosts = ['Homo sapiens', 'Felis catus', 'Mus musculus',
                     'Rattus norvegicus', 'Canis lupus familiaris',
                     'Ictidomys tridecemlineatus', 'Camelus dromedarius', 'Bos taurus', 'Pan troglodytes',
                     'Gallus gallus', 'Oryctolagus cuniculus', 'Equus caballus', 'Macaca mulatta', 'Ovis aries',
                     'Sus scrofa domesticus', 'Rhinolophus ferrumequinum', 'Mesocricetus auratus']

    # list of viruses
    list_of_viruses = ['Human coronavirus OC43', 'Human coronavirus HKU1',
                       'Middle East respiratory syndrome-related coronavirus',
                       'Severe acute respiratory syndrome coronavirus 2',
                       'Severe acute respiratory syndrome-related coronavirus', 'Human coronavirus 229E',
                       'Human coronavirus NL63']

    # path for content embeddings\
    content_emb_path = os.path.abspath('../data/embeddings/sentence_embedding/sentence_embedding.pkl')

    # path for constructed graph
    original_G_path = os.path.abspath('../data/link_prediction/data/original_G.txt')

    # if graph is not built, build the graph
    if not os.path.exists(original_G_path):
        print('NetworkX graph NOT built yet, building one now...')
        build_graph.build_graph(original_G_path=original_G_path, list_of_hosts=list_of_hosts,
                                list_of_viruses=list_of_viruses)
    else:
        print('NetworkX graph already built')

    # classification models
    # classification_models = ['Random Forest', 'K-Nearest-Neighbors', 'Support Vector Classification',
    #                          'Logistic Regression', 'Decision Tree', 'MLP']

    # path for model evaluation and comparison result

    # --------------------------------------------------------
    # ------------------------Execution-----------------------
    # --------------------------------------------------------
    # comparison
    model_iter = 10
    model_eval(structural_emb_path, content_emb_path, original_G_path, model_iter=model_iter)

    # model testing finished, now need to make predictions use node2vec_content
    model_pred(structural_emb_path, content_emb_path, original_G_path)
    print("\n-----------------END--------------------")


if __name__ == '__main__':
    main()
