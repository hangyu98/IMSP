import os

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from utils import filter_PPI_pred, filter_infection_pred, filter_unreliable_inf


class Classifier:
    def __init__(self, X_train, y_train, X_test, y_test, prediction_model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.prediction_model = prediction_model
        # error checking
        self.classifier = MLPClassifier(verbose=False)

    def train(self):

        self.classifier.max_iter = 1000
        print('In training', self.prediction_model, 'classifier')

        # fit the evaluation X and y into the model
        self.y_train = self.y_train.flatten()
        self.classifier.fit(self.X_train, self.y_train)

    def test_model(self):

        # extract the X_test from the test set
        y_pred = self.classifier.predict(self.X_train)
        # evaluation set performance
        print('Training set performance\n', classification_report(self.y_train, y_pred))

        # extract the X_test from the test set
        y_pred = self.classifier.predict(self.X_test)
        y_prob = self.classifier.predict_proba(self.X_test)

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

    def predict(self, full_G, all_selected_indices, index2pair_dict, binding, last_iter):

        # load np n-d array X from file
        X = np.loadtxt(os.path.abspath('data/classifier/X.txt'))
        print('In making predictions')
        prediction_prob = self.classifier.predict_proba(X)
        prediction = self.classifier.predict(X)

        X_list = list(X)

        for i in range(0, len(X_list)):
            pair = index2pair_dict[i]
            # if is a predicted link
            # if node type is different
            if i not in all_selected_indices:
                if not full_G.nodes[str(pair[0])]['type'] == full_G.nodes[str(pair[1])]['type']:
                    if prediction[i] == 2.0:
                        if full_G.has_edge(str(pair[0]), str(pair[1])):
                            pred_prob = full_G.get_edge_data(*(str(pair[0]), str(pair[1])))[
                                'probability_estimate']
                            new_pred_prob = (pred_prob + prediction_prob[i][2]) / 2.0
                            full_G.add_edge(str(pair[0]), str(pair[1]), etype='predicted',
                                            relation='infects',
                                            probability_estimate=new_pred_prob, connection='strong')
                        else:
                            full_G.add_edge(str(pair[0]), str(pair[1]), etype='predicted',
                                            relation='infects',
                                            probability_estimate=prediction_prob[i][2], connection='weak')
                    elif prediction[i] == 4.0:
                        if full_G.has_edge(str(pair[0]), str(pair[1])):
                            pred_prob = full_G.get_edge_data(*(str(pair[0]), str(pair[1])))[
                                'probability_estimate']
                            new_pred_prob = (pred_prob + prediction_prob[i][4]) / 2.0
                            full_G.add_edge(str(pair[0]), str(pair[1]), etype='predicted',
                                            relation='interacts',
                                            probability_estimate=new_pred_prob, connection='strong')
                        else:
                            full_G.add_edge(str(pair[0]), str(pair[1]), etype='predicted',
                                            relation='infects',
                                            probability_estimate=prediction_prob[i][4], connection='weak')
        print("Saving prediction data")
        filter_PPI_pred(full_G, edge_type='interacts', binding=binding)
        filter_infection_pred(full_G, edge_type='infects')
        if last_iter:
            filter_unreliable_inf(binding=binding)
        print('Prediction data saved!')
