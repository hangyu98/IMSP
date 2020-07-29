import pickle

import numpy as np
import os
from numpy import savetxt


def shrink_sampling(training_G, X, y, has_link_idx_lst, no_link_idx_lst, num_of_training_edges, num_of_test_edges,
                    index2pair_dict, all_selected_indices):
    """
    shrink the no_link_X
    :param all_selected_indices:
    :param index2pair_dict:
    :param X:
    :param y:
    :param training_G:
    :param no_link_idx_lst:
    :param has_link_idx_lst:
    :param num_of_training_edges:
    :param num_of_test_edges:
    :return:
    """
    X_train = np.empty([4 * num_of_training_edges, X.shape[1]])
    y_train = np.empty([4 * num_of_training_edges, 1])

    count = 0
    for idx in has_link_idx_lst:
        pair = index2pair_dict[idx]
        if (str(pair[0]), str(pair[1])) in training_G.edges():
            X_train[count] = X[idx]
            y_train[count] = y[idx]
            count = count + 1

    # selected_indices_train = np.random.choice(selected_indices, size=2 * num_of_training_edges, replace=False)
    # selected_no_link_X_train = no_link_X[selected_indices_train, :]

    X_negatives = np.empty([2 * num_of_test_edges, X.shape[1]])
    y_negatives = np.empty([2 * num_of_test_edges, 1])

    selected_indices = np.random.choice(len(no_link_idx_lst), size=2 * (num_of_training_edges + num_of_test_edges),
                                        replace=False)
    selected_indices_for_training = np.random.choice(selected_indices, size=2 * num_of_training_edges, replace=False)
    selected_indices_for_testing = np.setdiff1d(selected_indices, selected_indices_for_training)

    for idx in selected_indices_for_training:
        # print(idx)
        X_train[count] = X[no_link_idx_lst[idx]]
        y_train[count] = y[no_link_idx_lst[idx]]
        all_selected_indices.append(no_link_idx_lst[idx])
        count = count + 1

    count = 0
    for idx in selected_indices_for_testing:
        # print(idx)
        X_negatives[count] = X[no_link_idx_lst[idx]]
        y_negatives[count] = y[no_link_idx_lst[idx]]
        all_selected_indices.append(no_link_idx_lst[idx])
        count = count + 1

    # all_selected_indices = np.array(has_link_idx_lst.extend(list(selected_indices)))
    # save the files
    savetxt(os.path.abspath('../data/link_prediction/data/X_train.txt'), X_train)
    savetxt(os.path.abspath('../data/link_prediction/data/y_train.txt'), y_train)
    print('Sampled graph file saved')
    return X_train, y_train, X_negatives, y_negatives


# def enlarge_sampling(no_link_X, has_link_X, no_link_y, has_link_y):
#     """
#     shrink the no_link_X
#     :param no_link_X:
#     :param has_link_X:
#     :param no_link_y:
#     :param has_link_y:
#     :return:
#     """
#     size_of_no_link_X = no_link_X.shape[0]
#     size_of_has_link_X = has_link_X.shape[0]
#     num_to_select = size_of_no_link_X % size_of_has_link_X
#     num_to_copy = int(size_of_no_link_X / size_of_has_link_X)
#     print("need to select ", num_to_select, " elements from has_link_X")
#     # choose equal number of vectors from no_link_X as training set
#     selected_indices = np.random.choice(size_of_has_link_X, size=num_to_select, replace=False)
#     # stack the has_link_X with no_link_X as X_train]
#     enlarged_X = []
#     enlarged_y = []
#     for i in range(0, num_to_copy):
#         if len(enlarged_X) == 0:
#             enlarged_X = np.array(has_link_X)
#             enlarged_y = np.array(has_link_y)
#         else:
#             enlarged_X = np.vstack([enlarged_X, has_link_X])
#             enlarged_y = np.vstack([enlarged_y, has_link_y])
#
#     enlarged_X = np.vstack([enlarged_X, has_link_X[selected_indices, :]])
#     enlarged_y = np.vstack([enlarged_y, has_link_y[selected_indices, :]])
#     X_train = np.vstack([enlarged_X, no_link_X])
#     y_train = np.vstack([enlarged_y, no_link_y])
#     # save the files
#     savetxt(os.path.abspath('../data/link_prediction/X_train.txt'), X_train)
#     savetxt(os.path.abspath('../data/link_prediction/y_train.txt'), y_train)
#     print('Sampled graph file saved')
#     # normalize
#     return X_train, y_train


def random_sampling(training_G, X, y, y_copy, num_of_training_edges, num_of_test_edges, index2pair_dict):
    """
    make sure the size of samples with label 0 and label 1 are equal
    :param training_G:
    :param index2pair_dict:
    :param y_copy:
    :param num_of_test_edges:
    :param num_of_training_edges:
    :param X: original X containing all vectorized nodes
    :param y: related labels
    :return: X_train and y_train
    """
    all_selected_indices = []
    print("In randomly sampling...\n")
    no_link_idx_lst = []
    has_link_idx_lst = []
    for idx in range(0, len(y)):
        if y[idx] == 0:
            no_link_idx_lst.append(idx)
        elif y_copy[idx] >= 1.0:
            has_link_idx_lst.append(idx)
        if y[idx] >= 1:
            all_selected_indices.append(idx)

    # shrink the larger no_link_X to generate training set and test set
    X_train, y_train, X_negatives, y_negatives = shrink_sampling(training_G, X, y,
                                                                     has_link_idx_lst,
                                                                     no_link_idx_lst,
                                                                     num_of_training_edges,
                                                                     num_of_test_edges,
                                                                     index2pair_dict,
                                                                     all_selected_indices)

    print('\nData sampling finished')
    return X_train, y_train, X_negatives, y_negatives, all_selected_indices
