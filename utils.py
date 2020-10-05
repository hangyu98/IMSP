import copy
from random import choice
import os

import networkx as nx
import numpy as np
import csv
import sampling as sampling
from numpy import savetxt

from network.network import build_g
from network.network_data import list_of_hosts, list_of_viruses


# -------------------------------------------------------------------------------
# -----------------------------graph related-------------------------------------
# -------------------------------------------------------------------------------
def build_graph(bg):
    original_G_path = os.path.abspath('data/classifier/original_G.txt')
    # if network is not built, build the network
    if not os.path.exists(original_G_path) or bg:
        print('NetworkX network NOT built yet, building one now')
        build_g(original_G_path=original_G_path, list_of_hosts=list_of_hosts,
                list_of_viruses=list_of_viruses)
    else:
        print('NetworkX network already built')


def establish_training_G(G, re_build_g):
    training_G_path = os.path.abspath('data/classifier/training_G.txt')
    removed_edges_path = os.path.abspath('data/classifier/removed_edges.csv')

    total_similarity = 0
    total_belongs = 0
    total_infects = 0
    total_PPI = 0

    # count number of edges of different types
    for e in G.edges():
        edge_relation = G.get_edge_data(*e)['relation']
        if edge_relation.__contains__('similar'):
            total_similarity = total_similarity + 1
        elif edge_relation.__contains__('belongs'):
            total_belongs = total_belongs + 1
        elif edge_relation.__contains__('infects'):
            total_infects = total_infects + 1
        else:
            total_PPI = total_PPI + 1

    # target edges to be removed
    target_similarity = round(total_similarity / 5)
    target_belongs = round(total_belongs / 5)
    target_infects = round(total_infects / 5)
    target_PPI = round(total_PPI / 5)
    total_target = target_belongs + target_PPI + target_infects + target_similarity
    num_of_test_edges = total_target
    # if file is already there, do not split again. Instead, read network network from the file
    if os.path.exists(removed_edges_path) and not re_build_g:
        print('Removed edges found, now establishing evaluation network')
        removed_edges = []
        with open(removed_edges_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                removed_edges.append(row)
        # remove edges
        for edge in removed_edges:
            G.remove_edge(edge[0], edge[1])
        # save evaluation network
        nx.write_gml(G, training_G_path)
        # return training_G_path
        return training_G_path, num_of_test_edges
    # else, split into evaluation set and test set by a ratio of 8:2
    else:
        removed_edges = []
        print("Constructing evaluation G now removing 20% edges")
        while total_target > 0:
            edge = choice(list(G.edges(data=True)))
            # ensure connectivity
            H = copy.deepcopy(G)
            H.remove_edge(edge[0], edge[1])
            if not nx.is_connected(H):
                print('not connected, need to choose again')
                continue

            if edge[2]['relation'].__contains__('similar'):
                if target_similarity > 0:
                    target_similarity = target_similarity - 1
                else:
                    continue
            elif edge[2]['relation'].__contains__('belongs'):
                if target_belongs > 0:
                    target_belongs = target_belongs - 1
                else:
                    continue
            elif edge[2]['relation'].__contains__('infects'):
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
            total_target = total_target - 1
        print("Still connected?", nx.is_connected(G))
        # Save network
        print("Saving evaluation G to file")
        nx.write_gml(G, training_G_path)
        print("Saving removed edges to file")
        with open(removed_edges_path, 'w') as csv_file:
            for e in removed_edges:
                csv_file.write(str(e[0]) + ',' + str(e[1]) + '\n')

    return training_G_path, num_of_test_edges


def load_graph(training_G_path, structural_emb_path):
    new_G = nx.read_gml(training_G_path)

    structural_emb_dict = load_node_embeddings(structural_emb_path)

    return new_G, structural_emb_dict


# -------------------------------------------------------------------------------
# --------------------------classifier related-----------------------------------
# -------------------------------------------------------------------------------
def load_node_embeddings(emb_file_path):
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
    if content_emb_dict is not None:
        X = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1),
                      2 * (len(structural_emb_dict[1])) + len(content_emb_dict[(0, 1)])])
    else:
        X = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1),
                      2 * (len(structural_emb_dict[1]))])
    y = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1), 1])
    y_copy = np.empty([len(structural_emb_dict) * (len(structural_emb_dict) - 1), 1])
    count = 0
    for i in range(0, len(structural_emb_dict)):
        for j in range(0, len(structural_emb_dict)):
            if i != j:
                arr_i = np.array(structural_emb_dict[i])
                arr_j = np.array(structural_emb_dict[j])

                y_to_add = get_y_to_add(full_G, i, j)
                y_to_add_copy = get_y_to_add(training_G, i, j)

                # add embedding of the two nodes to represent edge
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
    savetxt(os.path.abspath('data/classifier/X.txt'), X)
    savetxt(os.path.abspath('data/classifier/y.txt'), y)
    savetxt(os.path.abspath('data/classifier/y_copy.txt'), y_copy)

    index2pair_dict = get_index2pair_dict(len(X), full_G)
    X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices \
        = sampling.random_sampling(training_G, X, y, y_copy,
                                   num_of_training_edges=len(training_G.edges()),
                                   num_of_test_edges=num_of_test_edges, index2pair_dict=index2pair_dict)

    return X_train, y_train, X_test_negatives, y_test_negatives, all_selected_indices, index2pair_dict


def get_y_to_add(full_G, i, j):
    if (str(i), str(j)) in full_G.edges():
        edge_relation = full_G.edges[str(i), str(j)]['relation']
        if edge_relation.__contains__('similar'):
            y_to_add = np.array([1])
        elif edge_relation.__contains__('infects'):
            y_to_add = np.array([2])
        elif edge_relation.__contains__('belongs'):
            y_to_add = np.array([3])
        else:
            y_to_add = np.array([4])
    else:
        y_to_add = np.array([0])
    return y_to_add


def get_index2pair_dict(length, G):
    index2pair_dict = {}
    src_node = 0
    dst_node = 0
    count = 0
    num_of_nodes = len(G.nodes())
    # fill in the dict
    while count < length:
        if src_node == dst_node:
            dst_node = (dst_node + 1) % num_of_nodes
            continue
        # add new node pair to the dict
        index2pair_dict[count] = (src_node, dst_node)
        dst_node = dst_node + 1
        if dst_node == num_of_nodes:
            src_node = src_node + 1
            dst_node = 0
        count = count + 1
    return index2pair_dict


def load_test_data(full_G, test_set_positives, structural_emb_dict, content_emb_dict, X_negatives, y_negatives):
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
            edge_relation = full_G.edges[str(i), str(j)]['relation']
            if edge_relation.__contains__('similar'):
                y_to_add = np.array([1])
            elif edge_relation.__contains__('infects'):
                y_to_add = np.array([2])
            elif edge_relation.__contains__('belongs'):
                y_to_add = np.array([3])
            else:
                y_to_add = np.array([4])
        else:
            y_to_add = np.array([0])
        # save
        y_positives[count] = y_to_add
        y_positives[count + 1] = y_to_add
        # increment count
        count = count + 2

    X_test = np.vstack([X_positives, X_negatives])
    y_test = np.concatenate([y_positives.flatten(), y_negatives.flatten()])

    savetxt(os.path.abspath('data/classifier/X_test.txt'), X_test)
    savetxt(os.path.abspath('data/classifier/y_test.txt'), y_test)

    return X_test, y_test


# -------------------------------------------------------------------------------
# -----------------------------save prediction-----------------------------------
# -------------------------------------------------------------------------------
def filter_PPI_pred(G, edge_type, binding):
    existed = remove_duplicates(edge_type)
    with open(os.path.abspath(
            'data/prediction/prediction_' + edge_type + '.csv'), 'a') as file:
        for e in G.edges():
            edge_data = G.get_edge_data(*e)
            if edge_data['relation'].__contains__('interacts') and edge_data['etype'] == 'predicted':
                src = str(e[0])
                dst = str(e[1])
                if not ((src, dst) in existed or (dst, src) in existed) and \
                        G.nodes[src]['group'] != G.nodes[dst]['group'] and \
                        'protein' in G.nodes[src]['group'] and 'protein' in G.nodes[dst]['group']:
                    basic_info = src + ',' + dst + ',' + G.nodes[src]['type'] + ' ' + \
                                 G.nodes[src]['host'] + ',' + G.nodes[dst]['type'] + ' ' + \
                                 G.nodes[dst]['host'] + ',' + str(edge_data['probability_estimate']) + ',' + \
                                 edge_data['connection'] + ','
                    if (G.nodes[src]['type'] == 'Spike' and
                        G.nodes[dst]['type'] == 'ACE2') or \
                            (G.nodes[src]['type'] == 'ACE2' and
                             G.nodes[dst]['type'] == 'Spike') or \
                            (G.nodes[src]['type'] == 'DPP4' and
                             G.nodes[dst]['type'] == 'Spike') or \
                            (G.nodes[src]['type'] == 'Spike' and
                             G.nodes[dst]['type'] == 'DPP4'):
                        to_write = basic_info + 'interacts' + ',' + 'reliable' + '\n'

                        binding.append((G.nodes[src]['host'] + ' ' + G.nodes[dst]['host']))
                        binding.append((G.nodes[dst]['host'] + ' ' + G.nodes[src]['host']))

                    elif G.nodes[src]['type'] == 'Spike' or G.nodes[dst]['type'] == 'Spike':
                        to_write = basic_info + 'interacts' + ',' + 'unreliable' + '\n'
                    else:
                        to_write = basic_info + 'interacts' + ',' + 'reliable' + '\n'
                    file.write(to_write)
        file.close()


def filter_infection_pred(G, edge_type):
    existed = remove_duplicates('temp_' + edge_type)
    with open(os.path.abspath(
            'data/prediction/prediction_temp_' + edge_type + '.csv'),
            'a') as file:
        for e in G.edges():
            edge_data = G.get_edge_data(*e)
            if edge_data['relation'].__contains__('infects') and edge_data['etype'] == 'predicted':
                src = str(e[0])
                dst = str(e[1])
                if not ((src, dst) in existed or (dst, src) in existed) and \
                        G.nodes[src]['group'] != G.nodes[dst]['group']:
                    if ((G.nodes[src]['type'] == 'virus' and G.nodes[dst]['type'] == 'host')
                            or (G.nodes[src]['type'] == 'host' and G.nodes[dst]['type'] == 'virus')):
                        basic_info = src + ',' + dst + ',' + \
                                     G.nodes[src]['type'] + ' ' + \
                                     G.nodes[src]['host'] + ',' + \
                                     G.nodes[dst]['type'] + ' ' + \
                                     G.nodes[dst]['host'] + ',' + \
                                     str(edge_data['probability_estimate']) + ',' + \
                                     edge_data['connection'] + ',' + \
                                     'infects'
                        basic_info = basic_info + '\n'
                        file.write(basic_info)
        file.close()


def filter_unreliable_inf(binding):
    with open(os.path.abspath('data/prediction/prediction_temp_infects.csv'), 'r') as read_csv, \
            open(os.path.abspath('data/prediction/prediction_infects.csv'), 'w') as write_csv:
        csv_reader = csv.reader(read_csv, delimiter=',')
        for row in csv_reader:
            virus = row[2].split(' ', 1)[1]
            host = row[3].split(' ', 1)[1]
            token = virus + ' ' + host
            if binding.__contains__(token):
                to_write = \
                    str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '').rsplit(',', 1)[0] \
                    + ',reliable' + '\n'
                write_csv.write(to_write)
            else:
                to_write = \
                    str(row).replace(', ', ',').replace('\'', '').replace('[', '').replace(']', '').rsplit(',', 1)[0] \
                    + ',unreliable' + '\n'
                write_csv.write(to_write)
        write_csv.close()
    os.remove(os.path.abspath('data/prediction/prediction_temp_infects.csv'))


def remove_duplicates(edge_type):
    existed = []
    # read all existed predictions --> eliminate duplicates
    if (os.path.exists(os.path.abspath(
            'data/prediction/prediction_' + edge_type + '.csv'))):
        with open(os.path.abspath(
                'data/prediction/prediction_' + edge_type + '.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                existed.append((str(row[0]), str(row[1])))
    return existed
