import numpy as np
import csv
from graph.build_graph_utils import add_hetero_edges


# # --------------------------------------------------------
# # --------------------Link Prediction---------------------
# # --------------------------------------------------------


def predict(G, structural_emb_path, hetero_edges, dict_of_nodes_groups):
    dict_for_matrix_X = {}

    matrix_X = []

    with open(structural_emb_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        next(csv_reader)
        count = 0
        for row in csv_reader:
            np_array = np.array([float(row[idx]) for idx in range(1, len(row))])  # extract vector
            vec_normalized = np_array / np.linalg.norm(np_array)  # normalize vector
            if len(matrix_X) == 0:  # first insertion into the matrix, set the proper dimension
                matrix_X = np.array(vec_normalized)
            else:
                matrix_X = np.vstack([matrix_X, vec_normalized])
            dict_for_matrix_X.update({count: int(row[0])})
            count = count + 1

    # calculate the similarity matrix using dot multiplication
    similarity_matrix = np.dot(matrix_X, np.transpose(matrix_X))

    # (src_idx, dst_idx, similarity_score)
    similarity_pair_list = []
    for row_idx in range(len(similarity_matrix)):
        for col_idx in range(0, row_idx):
            # get the id for nodes in the graph
            similarity_pair_list.append((dict_for_matrix_X[row_idx],
                                         dict_for_matrix_X[col_idx],
                                         similarity_matrix[row_idx][col_idx]))

    sorted_list = sorted(similarity_pair_list, key=lambda pair: pair[2], reverse=True)

    predicted_lst = []
    for ele in sorted_list:
        if (str(ele[0]), str(ele[1])) not in G.edges():
            predicted_lst.append(
                (G.nodes[str(ele[0])], G.nodes[str(ele[1])], {'similarity score': ele[2]})
            )

    # add predicted links
    for pred in predicted_lst[0: 200]:
        hetero_edges = hetero_edges + (add_hetero_edges(G=G,
                                                        dict_of_nodes_groups=dict_of_nodes_groups,
                                                        layer_1=pred[0]['layer'],
                                                        type_1=[pred[0]['type']],
                                                        host_list_1=[pred[0]['host']],
                                                        layer_2=pred[1]['layer'],
                                                        type_2=[pred[1]['type']],
                                                        host_list_2=[pred[1]['host']],
                                                        data=pred[2],
                                                        etype='predicted'))
        print("src node: ", pred[0]['disp'], "\ndst node: ", pred[1]['disp'], "\nsimilarity score: ",
              pred[2]['similarity score'], '\n')
