import os
import pickle

import networkx as nx
import numpy as np

from model import Classifier
from utils import establish_training_G, load_graph, load_training_data, load_test_data


def process(original_G_path, structural_emb_path, content_emb_path, model, pred):
    # --------------------------------------------------------
    # ------------------NETWORK CONSTRUCTION------------------
    # --------------------------------------------------------

    print("\n----------------------------------------")
    print("MODEL CONSTRUCTION")
    print("----------------------------------------")

    # read from file and re-establish a copy of the original graph
    full_G = nx.read_gml(original_G_path)
    print('# of full graph edges ', len(full_G.edges()))
    print('# of full graph nodes ', len(full_G.nodes()))

    print("---------------Step 1-------------------")
    print("In establishing training graph")

    training_G_path, num_of_test_edges = establish_training_G(full_G)

    print("Training graph completed")

    # load the training graph directly from file
    print("---------------Step 2-------------------")
    print("In loading training graph")
    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = load_graph(training_G_path,
                                                 structural_emb_path=structural_emb_path)
    file = open(content_emb_path, 'rb')
    # dump information to that file
    content_emb_dict = pickle.load(file)

    print("Training graph loaded")

    # get the training_X and labels y
    print("----------------Step 3------------------")
    print('In loading training graph')

    full_G = nx.read_gml(original_G_path)

    X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices, index2pair_dict \
        = load_training_data(full_G, training_G, structural_emb_dict, content_emb_dict,
                             num_of_test_edges)

    print("shape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    print("----------------Step 4------------------")
    print('In loading test graph')
    # test set is the set difference between edges in the original graph and the new graph
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = load_test_data(full_G,
                                    list(set(full_G.edges()) - set(training_G.edges())),
                                    structural_emb_dict, content_emb_dict,
                                    X_test_negatives, y_test_negatives)
    print("shape of X_test: ", X_test.shape)
    print("shape of y_test: ", y_test.shape)

    # --------------------------------------------------------
    # ------------------MODEL TRAINING -----------------------
    # --------------------------------------------------------

    print("----------------------------------------")
    print("CLASSIFICATION MODEL")
    print("----------------------------------------")
    if not pred:
        print("----------------------------------------")
        print('In model training')
        # get the prediction accuracy on training set

        # for model in classification_models:
        classifier_instance = Classifier(X_train, y_train, X_test, y_test, model)
        classifier_instance.train()
        accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = classifier_instance.test_model()
        return accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo
    else:
        print("----------------------------------------")
        print('In making predictions')
        print('Now combining X_train and X_test for full-graph training...')

        full_X_train = np.vstack([X_train, X_test])
        full_y_train = np.concatenate([y_train.flatten(), y_test])

        new_classifier_instance = Classifier(full_X_train, full_y_train, None, None, model)
        if not os.path.exists(os.path.abspath(
                '../data/prediction/result/prediction_infection_mat_' + new_classifier_instance.prediction_model + '.csv')):
            new_classifier_instance.train()
            new_classifier_instance.predict(full_G=full_G, all_selected_indices=all_selected_indices,
                                            index2pair_dict=index2pair_dict)
        else:
            new_classifier_instance.fdr(('infection', 'PPI'))


def process_comparison(original_G_path, structural_emb_path, model):
    # --------------------------------------------------------
    # --------------------Link prediction---------------------
    # --------------------------------------------------------
    print("----------------------------------------")
    print("MODEL CONSTRUCTION")
    print("----------------------------------------")
    # read from file and re-establish a copy of the original graph
    full_G = nx.read_gml(original_G_path)
    print('original num of edges: ', len(full_G.edges()))

    print("---------------Step 1-------------------")
    print("In establishing training graph")
    training_G_path, num_of_test_edges = establish_training_G(full_G)

    # load the training graph directly from file
    print("---------------Step 2-------------------")
    print("In loading training graph")
    # read the training_G from file and obtain a dict of {node: node_emb}
    training_G, structural_emb_dict = load_graph(training_G_path,
                                                 structural_emb_path=structural_emb_path)

    print("Training graph loaded")

    # get the training_X and labels y
    print("----------------Step 3------------------")
    print('In loading training data')

    full_G = nx.read_gml(original_G_path)

    X_train, y_train, X_test_to_add, y_test_to_add, all_selected_indices, index2pair_dict \
        = load_training_data(full_G, training_G, structural_emb_dict, None,
                             num_of_test_edges)

    print("shape of X_train: ", X_train.shape)
    print("shape of y_train: ", y_train.shape)

    print("----------------Step 4------------------")
    print('In loading test data')
    # test set is the set difference between edges in the original graph and the new graph
    full_G = nx.read_gml(original_G_path)
    # fill in the test set
    X_test, y_test = load_test_data(full_G,
                                    list(set(full_G.edges()) - set(training_G.edges())),
                                    structural_emb_dict, None,
                                    X_test_to_add, y_test_to_add)

    print("shape of X_test: ", X_test.shape)
    print("shape of y_test: ", y_test.shape)

    print("---------------Step 5-------------------")
    print('In testing classification model')
    # get the prediction accuracy on training set

    # for model in classification_models:
    classifier_instance = Classifier(X_train, y_train, X_test, y_test, model)
    classifier_instance.train()
    accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo = classifier_instance.test_model()
    print("-----------------END--------------------")

    return accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo


def model_eval(structural_emb_path, content_emb_path, original_G_path, model_iter):
    with open(os.path.abspath('../data/evaluation/comparison.csv'), 'w') as file:
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
                                       model='MLP')
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
            # write model performance to file after multiple iterations are complete
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

            emb_name = str(structural_emb_path[emb]).rsplit('/', 1)[1].split('.')[0] + '_structure with content'
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