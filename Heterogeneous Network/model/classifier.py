import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import csv
from numpy import genfromtxt


class Classifier:
    def __init__(self, X_train, y_train, X_test, y_test, prediction_model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.prediction_model = prediction_model
        # error checking
        if self.prediction_model == 'K-Nearest-Neighbors':
            self.classifier = KNeighborsClassifier()
        elif self.prediction_model == 'Logistic Regression':
            self.classifier = LogisticRegression()
        elif self.prediction_model == 'Support Vector Classification':
            self.classifier = SVC()
        elif self.prediction_model == 'Random Forest':
            self.classifier = RandomForestClassifier(n_jobs=-1)
        elif self.prediction_model == 'Decision Tree':
            self.classifier = DecisionTreeClassifier()
        elif self.prediction_model == 'MLP':
            self.classifier = MLPClassifier()
        else:
            print('error! ', self.prediction_model, ' not found!')

    def train(self):
        """
                Fit graph into a classification model and obtain prediction result
                :return: prediction accuracy
                """

        self.classifier.max_iter = 10000
        print('In training ', self.prediction_model, ' model')

        # fit the training X and y into the model
        self.y_train = self.y_train.flatten()
        self.classifier.fit(self.X_train, self.y_train)

    def test_model(self):

        # extract the X_test from the test set
        y_pred = self.classifier.predict(self.X_train)
        y_prob = self.classifier.predict_proba(self.X_train)
        with open(os.path.abspath('../data/link_prediction/result/prediction_train_' + self.prediction_model + '.csv'),
                  'a') as file:
            for i in range(0, len(self.X_test)):
                if y_pred[i] == 2.0 or y_pred[i] == 4.0:
                    to_write = str(y_prob[i][int(y_pred[i])]) + ',' + str(y_pred[i]) + '\n'
                    file.write(to_write)
            file.close()

        # training set performance
        print('Training set performance\n', classification_report(self.y_train, y_pred))

        # extract the X_test from the test set
        y_pred = self.classifier.predict(self.X_test)
        y_prob = self.classifier.predict_proba(self.X_test)
        with open(os.path.abspath('../data/link_prediction/result/prediction_test_' + self.prediction_model + '.csv'),
                  'a') as file:
            for i in range(0, len(self.X_test)):
                if y_pred[i] == 2.0 or y_pred[i] == 4.0:
                    to_write = str(y_prob[i][int(y_pred[i])]) + ',' + str(y_pred[i]) + '\n'
                    file.write(to_write)
            file.close()

        accuracy = self.classifier.score(self.X_test, self.y_test)

        report = classification_report(self.y_test, y_pred)

        print('Test set performance\n', report)

        macro_roc_auc_ovo = roc_auc_score(self.y_test, y_prob, multi_class="ovo",
                                          average="macro")
        weighted_roc_auc_ovo = roc_auc_score(self.y_test, y_prob, multi_class="ovo",
                                             average="weighted")
        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))

        report = classification_report(self.y_test, y_pred, output_dict=True)

        # save prediction...
        return accuracy, report, macro_roc_auc_ovo, weighted_roc_auc_ovo

    def predict(self, full_G, all_selected_indices, index2pair_dict):
        """
        save predicted nodes to file
        :param index2pair_dict:
        :param all_selected_indices:
        :param full_G: full graph G
        :return: null
        """
        # load np n-d array X from file
        X = np.loadtxt(os.path.abspath('../data/link_prediction/data/X.txt'))
        print('\nIn making predictions')
        prediction_prob = self.classifier.predict_proba(X)
        prediction = self.classifier.predict(X)

        X_list = list(X)

        self.save_pred(X_list, index2pair_dict, full_G, all_selected_indices, prediction, prediction_prob, 'infection')
        self.save_pred(X_list, index2pair_dict, full_G, all_selected_indices, prediction, prediction_prob, 'PPI')
        print('\nPredicted links saved!')

    def save_pred(self, X_list, index2pair_dict, full_G, all_selected_indices, prediction, prediction_prob, edge_type):
        if edge_type == 'infection':
            expected = 2.0
        elif edge_type == 'PPI':
            expected = 4.0
        confidences = []
        # generate matlab needed matrix
        with open(os.path.abspath(
                '../data/link_prediction/result/prediction_' + edge_type + '_mat_' + self.prediction_model + '.csv'),
                'w') as file:
            for i in range(0, len(X_list)):
                # if current idx is not in either training set or test set
                # extract edge representation --> node pairs
                pair = index2pair_dict[i]
                # if is a predicted link
                if i not in all_selected_indices:
                    # if type is different
                    if full_G.nodes[str(pair[0])]['type'] != full_G.nodes[str(pair[1])]['type']:
                        confidences.append(prediction_prob[i][int(expected)])
                        # p_value = st.norm.sf(abs(z_score)) * 2
                        file.write(str(i) + ',' + str(prediction_prob[i][int(expected)]) + '\n')
            file.close()

        # create temperate output file without q-values
        confidences = []
        with open(os.path.abspath(
                '../data/link_prediction/result/prediction_' + edge_type + '_temp_' + self.prediction_model + '.csv'),
                'w') as file:
            for i in range(0, len(X_list)):
                # if current idx is not in either training set or test set
                # extract edge representation --> node pairs
                pair = index2pair_dict[i]
                # if is a predicted link
                if i not in all_selected_indices:
                    if prediction[i] == expected:
                        # if type is different
                        if full_G.nodes[str(pair[0])]['type'] != full_G.nodes[str(pair[1])]['type']:
                            to_write = str(i) + ',' + full_G.nodes[str(pair[0])]['type'] + ' ' + \
                                       full_G.nodes[str(pair[0])]['host'] + \
                                       ',' + full_G.nodes[str(pair[1])]['type'] + ' ' + \
                                       full_G.nodes[str(pair[1])]['host'] + ',' + \
                                       str(prediction_prob[i][int(expected)]) + ','
                            confidences.append(prediction_prob[i][int(expected)])
                            to_write = to_write + edge_type + '\n'
                            file.write(to_write)
            file.close()

        # IMPORTANT: from here on, use MatLab code to generate q values

    def fdr(self, edge_types):
        """
        use storey's procedure to generate q values based on half-normal distribution
        :param edge_types: a tuple of edges
        :return: void
        """
        print('in FDR')
        for edge_type in edge_types:
            temp_dict = {}
            # open corrected file from matLab
            corrected_mat = genfromtxt(
                os.path.abspath(
                    '../data/link_prediction/result/prediction_' + edge_type + '_adjusted_' + self.prediction_model + '.csv'),
                delimiter=',')
            # transfer the matrix to a dict of {index: properties}
            for ele in corrected_mat:
                temp_dict[int(ele[0])] = ele[1:]
            with open(os.path.abspath(
                    '../data/link_prediction/result/prediction_' + edge_type + '_temp_' + self.prediction_model + '.csv'),
                    'r') as file:
                with open(os.path.abspath(
                        '../data/link_prediction/result/prediction_' + edge_type + '_' + self.prediction_model + '.csv'),
                        'w') as new_file:
                    csv_reader = csv.reader(file, delimiter=',')
                    for row in csv_reader:
                        idx = int(row[0])
                        to_write = ''
                        # copy existing elements in the row
                        for ele in row:
                            to_write += str(ele) + ','
                        q_val = float(temp_dict[idx][2])
                        # test hypothesis
                        if q_val < 0.01:
                            to_write = to_write + str(temp_dict[idx][2]) + ',True' + '\n'
                        else:
                            to_write = to_write + str(temp_dict[idx][2]) + ',False' + '\n'
                        new_file.write(to_write)
                    new_file.close()
                file.close()
