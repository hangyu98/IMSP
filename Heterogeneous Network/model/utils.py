import os as os
import pickle
from random import choice
import os

import networkx as nx
import numpy as np
import csv
import sampling
from numpy import savetxt

from classifier import Classifier


def establish_training_G(G):
    """
    remove 20% edges from the original graph, save the remaining graph to the test
    :param G: original G
    :return: file path of the training graph and number of removed edges
    """
    training_G_path = os.path.abspath('../data/link_prediction/data/training_G.txt')
    removed_edges_path = os.path.abspath('../data/link_prediction/data/removed_edges.csv')

    total_similarity = 0
    total_belongs = 0
    total_infects = 0
    total_PPI = 0

    # count number of edges of different types
    for e in G.edges():
        edge_data = G.get_edge_data(*e)['data']
        if edge_data.__contains__('similarity'):
            total_similarity = total_similarity + 1
        else:
            if G.get_edge_data(*e)['data']['relation'].__contains__('belongs'):
                total_belongs = total_belongs + 1
            elif G.get_edge_data(*e)['data']['relation'].__contains__('infects'):
                total_infects = total_infects + 1
            else:
                total_PPI = total_PPI + 1

    # target edges to be removed
    target_similarity = int(total_similarity / 5)
    target_belongs = int(total_belongs / 5)
    target_infects = int(total_infects / 5)
    target_PPI = int(total_PPI / 5)
    total_target = target_belongs + target_PPI + target_infects + target_similarity
    num_of_test_edges = total_target

    # if file is already there, do not split again. Instead, read graph graph from the file
    if os.path.exists(removed_edges_path):
        print('Removed edges found, now establishing training graph')
        removed_edges = []
        with open(removed_edges_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                removed_edges.append(row)
        # remove edges
        for edge in removed_edges:
            G.remove_edge(edge[0], edge[1])
        # save training graph
        nx.write_gml(G, training_G_path)
        # return training_G_path
        return training_G_path, num_of_test_edges
    # else, split into training set and test set by a ratio of 8:2
    else:
        removed_edges = []
        print("Removed edges NOT found, now removing 20% edges...")
        while total_target > 0:
            edge = choice(list(G.edges(data=True)))
            if edge[2]['data'].__contains__('similarity'):
                if target_similarity > 0:
                    target_similarity = target_similarity - 1
                else:
                    continue
            elif edge[2]['data']['relation'].__contains__('belongs'):
                if target_belongs > 0:
                    target_belongs = target_belongs - 1
                else:
                    continue
            elif edge[2]['data']['relation'].__contains__('infects'):
                if target_infects > 0:
                    target_infects = target_infects - 1
                else:
                    continue
            else:
                if target_PPI > 0:
                    target_PPI = target_PPI - 1
                else:
                    continue
            G.remove_edge(edge[0], edge[1])
            removed_edges.append((edge[0], edge[1]))
            # if add edge breaks graph connectivity, then add it back
            if not nx.is_connected(G):
                G.add_edge(edge[0], edge[1], data=edge[2]['data'], etype=edge[2]['etype'])
                if edge[2]['data'].__contains__('similarity'):
                    target_similarity = target_similarity + 1
                elif edge[2]['data']['relation'].__contains__('belongs'):
                    target_belongs = target_belongs + 1
                elif edge[2]['data']['relation'].__contains__('infects'):
                    target_infects = target_infects + 1
                else:
                    target_PPI = target_PPI + 1
                removed_edges.remove((edge[0], edge[1]))
                continue
            # if the graph is no longer connected, then do not update total target
            total_target = total_target - 1

        print("Still connected?", nx.is_connected(G))

        # Save graph
        print("Saving training graph to file...")
        nx.write_gml(G, training_G_path)
        print("Saving removed edges to file...")
        with open(removed_edges_path, 'w') as csv_file:
            for e in removed_edges:
                csv_file.write(str(e[0]) + ',' + str(e[1]) + '\n')

    return training_G_path, num_of_test_edges


def load_graph(training_G_path, structural_emb_path):
    """
    load the training graph from the path
    :param structural_emb_path:
    :param training_G_path: file path
    :return: training graph
    """
    # Read graph
    new_G = nx.read_gml(training_G_path)

    print("# of training graph nodes: ", len(new_G.nodes()))
    print("# of training graph edges: ", len(new_G.edges()))

    structural_emb_dict = load_node_embeddings(structural_emb_path)

    return new_G, structural_emb_dict


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


def load_training_data(full_G, training_G, structural_emb_dict, content_emb_dict, num_of_test_edges):
    """
    initialize the model by reading related files and
    :param full_G:
    :param training_G: training graph
    :param num_of_test_edges:
    :param content_emb_dict:
    :param structural_emb_dict:
    :return: sampled_X and sampled_y
    """
    if content_emb_dict is not None:
        X = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1),
                      2 * (len(structural_emb_dict[1])) + len(content_emb_dict[(0, 1)])])
    else:
        X = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1),
                      2 * (len(structural_emb_dict[1]))])
    y = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1), 1])
    y_copy = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1), 1])

    print('\nNeed to reconstruct X and y')
    count = 0
    for i in range(0, len(structural_emb_dict)):
        for j in range(0, len(structural_emb_dict)):
            if i != j:
                arr_i = np.array(structural_emb_dict[i])
                arr_j = np.array(structural_emb_dict[j])

                if (str(i), str(j)) in full_G.edges():
                    edge_data = full_G.edges[str(i), str(j)]['data']
                    if edge_data.__contains__('similarity'):
                        y_to_add = np.array([1])
                    else:
                        if edge_data['relation'].__contains__('infects'):
                            y_to_add = np.array([2])
                        elif edge_data['relation'].__contains__('belongs'):
                            y_to_add = np.array([3])
                        else:
                            y_to_add = np.array([4])
                else:
                    y_to_add = np.array([0])

                if (str(i), str(j)) in training_G.edges():
                    edge_data = training_G.edges[str(i), str(j)]['data']
                    if edge_data.__contains__('similarity'):
                        y_to_add_copy = np.array([1])
                    else:
                        if edge_data['relation'].__contains__('infects'):
                            y_to_add_copy = np.array([2])
                        elif edge_data['relation'].__contains__('belongs'):
                            y_to_add_copy = np.array([3])
                        else:
                            y_to_add_copy = np.array([4])
                else:
                    y_to_add_copy = np.array([0])
                # add embeddings of the two nodes to represent edge
                if content_emb_dict is not None:
                    arr_i_j = np.array(content_emb_dict[(i, j)])
                    edge_to_add = np.concatenate((arr_i, arr_j, arr_i_j))
                else:
                    edge_to_add = np.concatenate((arr_i, arr_j))
                X[count] = edge_to_add
                y[count] = y_to_add
                y_copy[count] = y_to_add_copy
                count = count + 1
    # save np matrix to file
    savetxt(os.path.abspath('../data/link_prediction/data/X.txt'), X)
    savetxt(os.path.abspath('../data/link_prediction/data/y.txt'), y)
    savetxt(os.path.abspath('../data/link_prediction/data/y_copy.txt'), y_copy)

    print("\nTraining graph loaded")
    print('X shape: ', X.shape)

    index2pair_dict = {}
    src_node = 0
    dst_node = 0
    count = 0
    num_of_nodes = len(full_G.nodes())
    # fill in the dict
    while count < len(X):
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

    X_train, y_train, X_negatives, y_negatives, all_selected_indices \
        = sampling.random_sampling(training_G, X, y, y_copy,
                                   num_of_training_edges=len(training_G.edges()),
                                   num_of_test_edges=num_of_test_edges, index2pair_dict=index2pair_dict)

    return X_train, y_train, X_negatives, y_negatives, all_selected_indices, index2pair_dict


def load_test_data(full_G, test_set_positives, structural_emb_dict, content_emb_dict, X_negatives, y_negatives):
    """
    extract test matrix from teh set of test graph
    :param content_emb_dict:
    :param full_G:
    :param y_negatives:
    :param X_negatives:
    :param test_set_positives:
    :param structural_emb_dict:
    :return: np matrix of X_test
    """
    size_of_positive_test_set = len(test_set_positives)

    if content_emb_dict is not None:
        X_positives = np.empty(
            [2 * size_of_positive_test_set, 2 * (len(structural_emb_dict[1])) + len(content_emb_dict[(0, 1)])])
    else:
        X_positives = np.empty(
            [2 * size_of_positive_test_set, 2 * (len(structural_emb_dict[1]))])

    y_positives = np.empty([2 * size_of_positive_test_set, 1])

    count = 0
    for pair in test_set_positives:
        i = int(pair[0])
        j = int(pair[1])

        # construct y_test
        arr_i = np.array(structural_emb_dict[i])
        arr_j = np.array(structural_emb_dict[j])
        if content_emb_dict is not None:
            arr_i_j = np.array(content_emb_dict[(i, j)])
            edge_to_add_1 = np.concatenate((arr_i, arr_j, arr_i_j))
            edge_to_add_2 = np.concatenate((arr_j, arr_i, arr_i_j))
        else:
            edge_to_add_1 = np.concatenate((arr_i, arr_j))
            edge_to_add_2 = np.concatenate((arr_j, arr_i))
        X_positives[count] = edge_to_add_1
        X_positives[count + 1] = edge_to_add_2

        # construct y_test
        if (str(i), str(j)) in full_G.edges():
            edge_data = full_G.edges[str(i), str(j)]['data']
            if edge_data.__contains__('similarity'):
                y_to_add = np.array([1])
            else:
                if edge_data['relation'].__contains__('infects'):
                    y_to_add = np.array([2])
                elif edge_data['relation'].__contains__('belongs'):
                    y_to_add = np.array([3])
                else:
                    y_to_add = np.array([4])
        else:
            print('no way!')
            y_to_add = np.array([0])
        # save
        y_positives[count] = y_to_add
        y_positives[count + 1] = y_to_add
        # increment count
        count = count + 2

    print('X_test_positives shape: ', X_positives.shape)
    print('X_test_negatives shape: ', X_negatives.shape)

    X_test = np.vstack([X_positives, X_negatives])
    y_test = np.concatenate([y_positives.flatten(), y_negatives.flatten()])

    savetxt(os.path.abspath('../data/link_prediction/data/X_test.txt'), X_test)
    savetxt(os.path.abspath('../data/link_prediction/data/y_test.txt'), y_test)

    print('\ntest data loaded')

    return X_test, y_test


def process(original_G_path, structural_emb_path, content_emb_path, model, pred):
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

    training_G_path, num_of_test_edges = establish_training_G(full_G)

    print("\nTraining graph completed")

    # load the training graph directly from file
    print("\n---------------Step 2-------------------")
    print("In loading training graph")
    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = load_graph(training_G_path,
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
        = load_training_data(full_G, training_G, structural_emb_dict, content_emb_dict,
                             num_of_test_edges)

    print("\nshape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    print("\n----------------Step 4------------------")
    print('In loading test graph')
    # test set is the set difference between edges in the original graph and the new graph
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = load_test_data(full_G,
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
    else:
        print("\n----------------------------------------")
        print('In making predictions\n')
        print('Now combining X_train and X_test for full-graph training...\n')

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


def process_comparison(original_G_path, structural_emb_path, model):
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
    training_G_path, num_of_test_edges = establish_training_G(full_G)

    # load the training graph directly from file
    print("\n---------------Step 2-------------------")
    print("In loading training graph")
    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = load_graph(training_G_path,
                                                 structural_emb_path=structural_emb_path)

    print("\nTraining graph loaded")

    # get the training_X and labels y
    print("\n----------------Step 3------------------")
    print('In loading training data')

    full_G = nx.read_gml(original_G_path)

    X_train, y_train, X_test_to_add, y_test_to_add, all_selected_indices, index2pair_dict \
        = load_training_data(full_G, training_G, structural_emb_dict, None,
                             num_of_test_edges)

    print("\nshape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    print("\n----------------Step 4------------------")
    print('In loading test data')
    # test set is the set difference between edges in the original graph and the new graph
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = load_test_data(full_G,
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


def model_eval(structural_emb_path, content_emb_path, original_G_path, model_iter):
    with open(os.path.abspath('../data/model_evaluation/comparison.csv'), 'w') as file:
        file.write(
            'embedding model,accuracy,infection precision,infection recall,infection f1-score,PPI precision,'
            'PPI recall,PPI f1-score,weighted precision,weighted recall,weighted f1-score,ROC macro,ROC weighted\n'
        )
        for emb in range(len(structural_emb_path)):
            print('Testing with only node structural embedding: ', structural_emb_path[emb])
            acc_result_lst = []
            infection_precision_result_lst = []
            PPI_precision_result_lst = []
            infection_recall_result_lst = []
            PPI_recall_result_lst = []
            infection_f1_result_lst = []
            PPI_f1_result_lst = []
            overall_precision_result_lst = []
            overall_recall_result_lst = []
            overall_f1_result_lst = []
            ROC_score_macro = []
            ROC_score_weighted = []

            for i in range(0, model_iter):
                acc, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = \
                    process_comparison(original_G_path=original_G_path,
                                       structural_emb_path=
                                       structural_emb_path[emb],
                                       model='MLP',
                                       re_sample=True)
                acc_result_lst.append(acc)
                infection_precision_result_lst.append(report['2.0']['precision'])
                PPI_precision_result_lst.append(report['4.0']['precision'])
                infection_recall_result_lst.append(report['2.0']['recall'])
                PPI_recall_result_lst.append(report['4.0']['recall'])
                infection_f1_result_lst.append(report['2.0']['f1-score'])
                PPI_f1_result_lst.append(report['4.0']['f1-score'])
                overall_precision_result_lst.append(report['weighted avg']['precision'])
                overall_recall_result_lst.append(report['weighted avg']['recall'])
                overall_f1_result_lst.append(report['weighted avg']['f1-score'])
                ROC_score_macro.append(macro_roc_auc_ovo)
                ROC_score_weighted.append(weighted_roc_auc_ovo)

            emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0] + '_structure only'
            # write model performance to file
            to_write = emb_name + ',' + str(np.mean(acc_result_lst)) + ',' + str(
                np.mean(infection_precision_result_lst)) + ',' + str(
                np.mean(infection_recall_result_lst)) + ',' + str(
                np.mean(infection_f1_result_lst)) + ',' + str(
                np.mean(PPI_precision_result_lst)) + ',' + str(np.mean(PPI_recall_result_lst)) + ',' + str(
                np.mean(PPI_f1_result_lst)) + ',' + str(np.mean(overall_precision_result_lst)) + ',' + str(
                np.mean(overall_recall_result_lst)) + ',' + str(np.mean(overall_f1_result_lst)) + ',' + str(
                np.mean(ROC_score_macro)) + ',' + str(np.mean(ROC_score_weighted)) + '\n'
            print(to_write)
            file.write(to_write)

            print('Testing with content embedding combined with structural embedding: ', structural_emb_path[emb])

            acc_result_lst = []
            infection_precision_result_lst = []
            PPI_precision_result_lst = []
            infection_recall_result_lst = []
            PPI_recall_result_lst = []
            infection_f1_result_lst = []
            PPI_f1_result_lst = []
            overall_precision_result_lst = []
            overall_recall_result_lst = []
            overall_f1_result_lst = []
            ROC_score_macro = []
            ROC_score_weighted = []

            for i in range(0, model_iter):
                acc, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = \
                    process(original_G_path=original_G_path,
                            structural_emb_path=
                            structural_emb_path[2],
                            content_emb_path=content_emb_path,
                            model='MLP',
                            re_sample=True,
                            pred=False)
                acc_result_lst.append(acc)
                infection_precision_result_lst.append(report['2.0']['precision'])
                PPI_precision_result_lst.append(report['4.0']['precision'])
                infection_recall_result_lst.append(report['2.0']['recall'])
                PPI_recall_result_lst.append(report['4.0']['recall'])
                infection_f1_result_lst.append(report['2.0']['f1-score'])
                PPI_f1_result_lst.append(report['4.0']['f1-score'])
                overall_precision_result_lst.append(report['weighted avg']['precision'])
                overall_recall_result_lst.append(report['weighted avg']['recall'])
                overall_f1_result_lst.append(report['weighted avg']['f1-score'])
                ROC_score_macro.append(macro_roc_auc_ovo)
                ROC_score_weighted.append(weighted_roc_auc_ovo)

            emb_name = str(structural_emb_path[2]).rsplit('/', 1)[1].split('.')[0] + '_structure with content'
            # write model performance to file
            to_write = emb_name + ',' + str(np.mean(acc_result_lst)) + ',' + str(
                np.mean(infection_precision_result_lst)) + ',' + str(
                np.mean(infection_recall_result_lst)) + ',' + str(
                np.mean(infection_f1_result_lst)) + ',' + str(
                np.mean(PPI_precision_result_lst)) + ',' + str(np.mean(PPI_recall_result_lst)) + ',' + str(
                np.mean(PPI_f1_result_lst)) + ',' + str(np.mean(overall_precision_result_lst)) + ',' + str(
                np.mean(overall_recall_result_lst)) + ',' + str(np.mean(overall_f1_result_lst)) + ',' + str(
                np.mean(ROC_score_macro)) + ',' + str(np.mean(ROC_score_weighted)) + '\n'
            # print(to_write)
            file.write(to_write)
        file.close()

def model_pred(structural_emb_path, content_emb_path, original_G_path):
    process(original_G_path=original_G_path, structural_emb_path=structural_emb_path[2],
            content_emb_path=content_emb_path, model='MLP', pred=True)