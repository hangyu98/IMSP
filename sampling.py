import numpy as np
import os
from numpy import savetxt

import utils as utils


def random_sampling_helper(training_G, X, y, positives, negatives, num_of_training_edges,
                           num_of_test_edges, index2pair_dict, all_selected_indices):

    print('Sampling negatives')
    X_train = np.empty([4 * num_of_training_edges, X.shape[1]])
    y_train = np.empty([4 * num_of_training_edges, 1])

    # positive training sample
    count = 0
    for idx in positives:
        pair = index2pair_dict[idx]
        if (str(pair[0]), str(pair[1])) in training_G.edges():
            X_train[count] = X[idx]
            y_train[count] = y[idx]
            count = count + 1

    # negative samples
    X_test_negatives = np.empty([2 * num_of_test_edges, X.shape[1]])
    y_test_negatives = np.empty([2 * num_of_test_edges, 1])

    print("size of all negatives: ", len(negatives))
    known_negatives = get_known_negatives(training_G, X)
    print("size of known negatives: ", len(known_negatives))
    unknown_negatives = np.setdiff1d(negatives, known_negatives)
    print("size of unknown negatives: ", len(unknown_negatives))

    known_negatives_testing = np.random.choice(known_negatives,
                                               size=int(len(known_negatives) * 0.2),
                                               replace=False)

    known_negatives_training = np.setdiff1d(known_negatives, known_negatives_testing)

    selected_unknown_negatives = np.random.choice(unknown_negatives,
                                                  size=2 * (num_of_training_edges + num_of_test_edges) - len(
                                                      known_negatives),
                                                  replace=False)

    unknown_negatives_training = np.random.choice(selected_unknown_negatives,
                                                  size=2 * num_of_training_edges - (
                                                          len(known_negatives) - int(len(known_negatives) * 0.2)),
                                                  replace=False)

    unknown_negatives_testing = np.setdiff1d(selected_unknown_negatives, unknown_negatives_training)

    selected_indices_for_training = np.concatenate((known_negatives_training, unknown_negatives_training))

    selected_indices_for_testing = np.concatenate((known_negatives_testing, unknown_negatives_testing))

    for idx in selected_indices_for_training:
        X_train[count] = X[idx]
        y_train[count] = y[idx]
        all_selected_indices.append(idx)
        count = count + 1

    count = 0
    for idx in selected_indices_for_testing:
        X_test_negatives[count] = X[idx]
        y_test_negatives[count] = y[idx]
        all_selected_indices.append(idx)
        count = count + 1

    # save the files
    savetxt(os.path.abspath('data/classifier/X_train.txt'), X_train)
    savetxt(os.path.abspath('data/classifier/y_train.txt'), y_train)
    return X_train, y_train, X_test_negatives, y_test_negatives


def get_known_negatives(training_G, X):
    known_negatives = []

    mers = "Middle East respiratory syndrome-related coronavirus"
    sars = "Severe acute respiratory syndrome-related coronavirus"
    sars2 = "Severe acute respiratory syndrome coronavirus 2"
    nl63 = "Human coronavirus NL63"
    ACE2s = [sars, sars2, nl63]

    index2pair_dict = utils.get_index2pair_dict(len(X), training_G)
    for key in index2pair_dict:
        ele = index2pair_dict[key]
        src_node = str(ele[0])
        dst_node = str(ele[1])
        if (training_G.nodes[src_node]['type'] == 'Spike'
            and training_G.nodes[src_node]['host'] == mers
            and training_G.nodes[dst_node]['type'] == 'ACE2') or \
                (training_G.nodes[src_node]['type'] == 'Spike'
                 and training_G.nodes[src_node]['host'] in ACE2s
                 and training_G.nodes[dst_node]['type'] == 'DPP4') or \
                (training_G.nodes[dst_node]['type'] == 'Spike'
                 and training_G.nodes[dst_node]['host'] == mers
                 and training_G.nodes[src_node]['type'] == 'ACE2') or \
                (training_G.nodes[dst_node]['type'] == 'Spike'
                 and training_G.nodes[dst_node]['host'] in ACE2s
                 and training_G.nodes[src_node]['type'] == 'DPP4'):
            known_negatives.append(key)
    # print(len(known_negatives))
    return known_negatives


def random_sampling(training_G, X, y, y_copy, num_of_training_edges, num_of_test_edges, index2pair_dict):
    all_selected_indices = []
    negative_idx_lst = []
    positive_idx_lst = []
    for idx in range(0, len(X)):
        if y[idx] == 0:
            negative_idx_lst.append(idx)
        elif y_copy[idx] >= 1:
            positive_idx_lst.append(idx)
        if y[idx] >= 1:
            all_selected_indices.append(idx)

    # shrink the larger negative_X to generate evaluation set and test set
    X_train, y_train, X_test_negatives, y_test_negatives = random_sampling_helper(training_G, X, y,
                                                                                  positive_idx_lst,
                                                                                  negative_idx_lst,
                                                                                  num_of_training_edges,
                                                                                  num_of_test_edges,
                                                                                  index2pair_dict,
                                                                                  all_selected_indices)

    return X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices
