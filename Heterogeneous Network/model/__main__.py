"""import modules"""

import networkx as nx
import os as os
from classifier import Classifier
from graph import build_graph
import numpy as np

import utils
import pickle


def main():
    # --------------------------------------------------------
    # --------------------Configurations----------------------
    # --------------------------------------------------------

    # paths for structural embeddings
    structural_emb_path = [
        os.path.abspath('../data/embeddings/deepwalk_embedding/deepwalk_unweighted_128.csv'),
        os.path.abspath('../data/embeddings/node2vec_embedding/node2vec_structural_128.csv'),
        os.path.abspath('../data/embeddings/node2vec_embedding/node2vec_content_128.csv'),
        os.path.abspath('../data/embeddings/node2vec_embedding/node2vec_unweighted_128.csv'),
        os.path.abspath('../data/embeddings/line_embedding/line_embedding_128.csv'),
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

    # iter
    model_iter = 5
    # run the test multiple times and get an average
    # with open(os.path.abspath('../data/model_evaluation/comparison.csv'), 'w') as file:
    #     file.write(
    #         'embedding model,accuracy,infection precision,infection recall,infection f1-score,PPI precision,'
    #         'PPI recall,PPI f1-score,weighted precision,weighted recall,weighted f1-score,ROC macro,ROC weighted\n'
    #     )
    #     for emb in range(len(structural_emb_path)):
    #         print('Testing with only node structural embedding: ', structural_emb_path[emb])
    #         acc_result_lst = []
    #         infection_precision_result_lst = []
    #         PPI_precision_result_lst = []
    #         infection_recall_result_lst = []
    #         PPI_recall_result_lst = []
    #         infection_f1_result_lst = []
    #         PPI_f1_result_lst = []
    #         overall_precision_result_lst = []
    #         overall_recall_result_lst = []
    #         overall_f1_result_lst = []
    #         ROC_score_macro = []
    #         ROC_score_weighted = []
    #
    #         for i in range(0, model_iter):
    #             acc, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = \
    #                 process_comparison(original_G_path=original_G_path,
    #                                    structural_emb_path=
    #                                    structural_emb_path[emb],
    #                                    model='MLP',
    #                                    re_sample=True)
    #             acc_result_lst.append(acc)
    #             infection_precision_result_lst.append(report['2.0']['precision'])
    #             PPI_precision_result_lst.append(report['4.0']['precision'])
    #             infection_recall_result_lst.append(report['2.0']['recall'])
    #             PPI_recall_result_lst.append(report['4.0']['recall'])
    #             infection_f1_result_lst.append(report['2.0']['f1-score'])
    #             PPI_f1_result_lst.append(report['4.0']['f1-score'])
    #             overall_precision_result_lst.append(report['weighted avg']['precision'])
    #             overall_recall_result_lst.append(report['weighted avg']['recall'])
    #             overall_f1_result_lst.append(report['weighted avg']['f1-score'])
    #             ROC_score_macro.append(macro_roc_auc_ovo)
    #             ROC_score_weighted.append(weighted_roc_auc_ovo)
    #
    #         emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0]
    #         # write model performance to file
    #         to_write = emb_name + ',' + str(np.mean(acc_result_lst)) + ',' + str(
    #             np.mean(infection_precision_result_lst)) + ',' + str(
    #             np.mean(infection_recall_result_lst)) + ',' + str(
    #             np.mean(infection_f1_result_lst)) + ',' + str(
    #             np.mean(PPI_precision_result_lst)) + ',' + str(np.mean(PPI_recall_result_lst)) + ',' + str(
    #             np.mean(PPI_f1_result_lst)) + ',' + str(np.mean(overall_precision_result_lst)) + ',' + str(
    #             np.mean(overall_recall_result_lst)) + ',' + str(np.mean(overall_f1_result_lst)) + ',' + str(
    #             np.mean(ROC_score_macro)) + ',' + str(np.mean(ROC_score_weighted)) + '\n'
    #         print(to_write)
    #         file.write(to_write)
    #
    #         print('Testing with content embedding combined with structural embedding: ', structural_emb_path[emb])
    #         acc_result_lst = []
    #         infection_precision_result_lst = []
    #         PPI_precision_result_lst = []
    #         infection_recall_result_lst = []
    #         PPI_recall_result_lst = []
    #         infection_f1_result_lst = []
    #         PPI_f1_result_lst = []
    #         overall_precision_result_lst = []
    #         overall_recall_result_lst = []
    #         overall_f1_result_lst = []
    #         ROC_score_macro = []
    #         ROC_score_weighted = []
    #
    #         for i in range(0, model_iter):
    #             acc, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = \
    #                 process(original_G_path=original_G_path,
    #                         structural_emb_path=
    #                         structural_emb_path[2],
    #                         content_emb_path=content_emb_path,
    #                         model='MLP',
    #                         re_sample=True,
    #                         pred=False)
    #             acc_result_lst.append(acc)
    #             infection_precision_result_lst.append(report['2.0']['precision'])
    #             PPI_precision_result_lst.append(report['4.0']['precision'])
    #             infection_recall_result_lst.append(report['2.0']['recall'])
    #             PPI_recall_result_lst.append(report['4.0']['recall'])
    #             infection_f1_result_lst.append(report['2.0']['f1-score'])
    #             PPI_f1_result_lst.append(report['4.0']['f1-score'])
    #             overall_precision_result_lst.append(report['weighted avg']['precision'])
    #             overall_recall_result_lst.append(report['weighted avg']['recall'])
    #             overall_f1_result_lst.append(report['weighted avg']['f1-score'])
    #             ROC_score_macro.append(macro_roc_auc_ovo)
    #             ROC_score_weighted.append(weighted_roc_auc_ovo)
    #
    #         emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0]
    #         # write model performance to file
    #         to_write = emb_name + ',' + str(np.mean(acc_result_lst)) + ',' + str(
    #             np.mean(infection_precision_result_lst)) + ',' + str(
    #             np.mean(infection_recall_result_lst)) + ',' + str(
    #             np.mean(infection_f1_result_lst)) + ',' + str(
    #             np.mean(PPI_precision_result_lst)) + ',' + str(np.mean(PPI_recall_result_lst)) + ',' + str(
    #             np.mean(PPI_f1_result_lst)) + ',' + str(np.mean(overall_precision_result_lst)) + ',' + str(
    #             np.mean(overall_recall_result_lst)) + ',' + str(np.mean(overall_f1_result_lst)) + ',' + str(
    #             np.mean(ROC_score_macro)) + ',' + str(np.mean(ROC_score_weighted)) + '\n'
    #         print(to_write)
    #         file.write(to_write)
    #     file.close()

    # model testing finished, now need to make predictions use node2vec_content
    process(original_G_path=original_G_path, structural_emb_path=structural_emb_path[2],
            content_emb_path=content_emb_path, model='MLP', re_sample=True, pred=True)


def process(original_G_path, structural_emb_path, content_emb_path, model, re_sample, pred, ):
    # --------------------------------------------------------
    # ------------------NETWORK CONSTRUCTION------------------
    # --------------------------------------------------------

    print("\n\n----------------------------------------")
    print("MODEL CONSTRUCTION")
    print("----------------------------------------")

    # read from file and re-establish a copy of the original graph
    full_G = nx.read_gml(original_G_path)
    print('# of full graph edges ', len(full_G.edges()))
    print('# of full graph nodes ', len(full_G.nodes()))

    print("\n---------------Step 1-------------------")
    print("In establishing training graph")

    training_G_path, num_of_test_edges = utils.establish_training_G(full_G)

    print("\nTraining graph completed")

    # load the training graph directly from file
    print("\n---------------Step 2-------------------")
    print("In loading training graph")
    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = utils.load_graph(training_G_path,
                                                       structural_emb_path=structural_emb_path)
    file = open(content_emb_path, 'rb')
    # dump information to that file
    content_emb_dict = pickle.load(file)

    print("\nTraining graph loaded")

    # get the training_X and labels y
    print("\n----------------Step 3------------------")
    print('In loading training graph')

    full_G = nx.read_gml(original_G_path)

    X_train, y_train, X_negatives, y_negatives, all_selected_indices, index2pair_dict \
        = utils.load_training_data(full_G, training_G, structural_emb_dict, content_emb_dict,
                                   num_of_test_edges,
                                   reset_X=re_sample)

    print("\nshape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    print("\n----------------Step 4------------------")
    print('In loading test graph')
    # test set is the set difference between edges in the original graph and the new graph
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = utils.load_test_data(full_G,
                                          list(set(full_G.edges()) - set(training_G.edges())),
                                          structural_emb_dict, content_emb_dict,
                                          X_negatives, y_negatives)
    print("\nshape of X_test: ", X_test.shape)
    print("shape of y_test: ", y_test.shape)

    # --------------------------------------------------------
    # ------------------MODEL TRAINING -----------------------
    # --------------------------------------------------------

    print("\n\n----------------------------------------")
    print("CLASSIFICATION MODEL")
    print("----------------------------------------")
    if not pred:
        print("\n----------------------------------------")
        print('In model training\n')
        # get the prediction accuracy on training set

        # for model in classification_models:
        classifier_instance = Classifier(X_train, y_train, X_test, y_test, model)
        classifier_instance.train()
        accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = classifier_instance.test_model()
        return accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo

    if pred:
        print("\n----------------------------------------")
        print('In making predictions\n')
        full_X_train = np.vstack([X_train, X_test])
        full_y_train = np.concatenate([y_train.flatten(), y_test])

        new_classifier_instance = Classifier(full_X_train, full_y_train, None, None, model)
        if not os.path.exists(os.path.abspath(os.path.abspath(
                '../data/link_prediction/result/prediction_infection_mat_' + new_classifier_instance.prediction_model + '.csv'))):
            new_classifier_instance.train()
            new_classifier_instance.predict(full_G=full_G, all_selected_indices=all_selected_indices,
                                            index2pair_dict=index2pair_dict)
        else:
            new_classifier_instance.fdr(('infection', 'PPI'))
    print("\n-----------------END--------------------")


def process_comparison(original_G_path, structural_emb_path, model, re_sample):
    # --------------------------------------------------------
    # --------------------Link prediction---------------------
    # --------------------------------------------------------
    print("\n\n----------------------------------------")
    print("MODEL CONSTRUCTION")
    print("----------------------------------------")
    # read from file and re-establish a copy of the original graph
    full_G = nx.read_gml(original_G_path)
    print('original num of edges: ', len(full_G.edges()))

    print("\n---------------Step 1-------------------")
    print("In establishing training graph")
    training_G_path, num_of_test_edges = utils.establish_training_G(full_G)

    # load the training graph directly from file
    print("\n---------------Step 2-------------------")
    print("In loading training graph")
    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = utils.load_graph(training_G_path,
                                                       structural_emb_path=structural_emb_path)

    print("\nTraining graph loaded")

    # get the training_X and labels y
    print("\n----------------Step 3------------------")
    print('In loading training data')

    full_G = nx.read_gml(original_G_path)

    X_train, y_train, X_test_to_add, y_test_to_add, all_selected_indices, index2pair_dict \
        = utils.load_training_data(full_G, training_G, structural_emb_dict, None,
                                   num_of_test_edges,
                                   reset_X=re_sample)

    print("\nshape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    print("\n----------------Step 4------------------")
    print('In loading test data')
    # test set is the set difference between edges in the original graph and the new graph
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = utils.load_test_data(full_G,
                                          list(set(full_G.edges()) - set(training_G.edges())),
                                          structural_emb_dict, None,
                                          X_test_to_add, y_test_to_add, )
    print("\nshape of X_test: ", X_test.shape)
    print("shape of y_test: ", y_test.shape)

    print("\n---------------Step 5-------------------")
    print('In testing classification model\n')
    # get the prediction accuracy on training set

    # for model in classification_models:
    classifier_instance = Classifier(X_train, y_train, X_test, y_test, model)
    classifier_instance.train()
    accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = classifier_instance.test_model()
    print("\n-----------------END--------------------")

    return accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo


if __name__ == '__main__':
    main()
