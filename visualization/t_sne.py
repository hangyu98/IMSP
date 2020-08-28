from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def t_sne(structural_emb_path):
    node_targets = list(range(233))
    matrix_X = []
    with open(structural_emb_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        next(csv_reader)
        for row in csv_reader:
            np_array = np.array([float(row[idx]) for idx in range(1, len(row))])
            # vec_normalized = np_array / np.linalg.norm(np_array)
            vec_normalized = np_array
            if len(matrix_X) == 0:
                matrix_X = np.array(vec_normalized)
            else:
                matrix_X = np.vstack([matrix_X, vec_normalized])

    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(matrix_X)
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(10, 8))
    plt.scatter(node_embeddings_2d[:, 0],
                node_embeddings_2d[:, 1],
                c=node_colours, cmap="jet", alpha=0.7)
    plt.show()


def t_sne_2(training_X_path):
    node_targets = list(range(3664 * 2))
    matrix_X = np.loadtxt(training_X_path)

    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(matrix_X)
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(20, 20))
    plt.scatter(node_embeddings_2d[:, 0],
                node_embeddings_2d[:, 1],
                c=node_colours, cmap="jet", alpha=0.7)
    plt.show()


def t_sne_3(training_X_path):
    node_targets = list(range(192))
    matrix_X = []

    with open(structural_emb_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        next(csv_reader)
        for row in csv_reader:
            np_array = np.array([float(row[idx]) for idx in range(1, len(row))])
            # vec_normalized = np_array / np.linalg.norm(np_array)
            vec_normalized = np_array
            if len(matrix_X) == 0:
                matrix_X = np.array(vec_normalized)
            else:
                matrix_X = np.vstack([matrix_X, vec_normalized])

    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings_2d = tsne.fit_transform(matrix_X)
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(10, 8))
    plt.scatter(node_embeddings_2d[:, 0],
                node_embeddings_2d[:, 1],
                c=node_colours, cmap="jet", alpha=0.7)
    plt.show()


if __name__ == '__main__':
    # print(os.path.abspath('../data/prediction/node2vec_128.csv'))
    structural_emb_path = os.path.abspath('../data/embedding_result/node2vec_embedding/node2vec_128.csv')
    t_sne(structural_emb_path=structural_emb_path)

    # training_X_path = os.path.abspath('../data/prediction/X_train.txt')
    # t_sne_2(training_X_path=training_X_path)
