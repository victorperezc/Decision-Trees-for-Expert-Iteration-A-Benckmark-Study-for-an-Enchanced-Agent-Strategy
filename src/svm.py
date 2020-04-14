import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from src.definitions import RESULTS_DIR

class SVM():

    def __init__(self,dataset,features,predictor,classes,name,cm=False,vi=False,dump_regressor=False):
        self.data = dataset
        self.features = features
        self.predictor = predictor
        self.classes = classes
        self.name = name
        self.cm = cm
        self.vi = vi
        self.dump_regressor = dump_regressor
        self.classifier = SVC(kernel='rbf')

    # Train the Random Forest Classifier
    def train(self):
        train_features, test_features, train_labels, test_labels = self.split(self.data)
        self.classifier.fit(train_features, train_labels)
        return self.accuracy(test_features,test_labels)

    def test(self,features,labels):
        predictions = self.classifier.predict(features)
        probs = self.classifier.predict_proba(features)[:, 1]
        return predictions,probs
        
    def accuracy(self,predictions,labels):
        return self.classifier.score(predictions, labels)

    def predict(self,d):
        return self.classifier.predict(d)[0]

    def split(self,data):
        labels = np.array(data[self.predictor])
        labels = [int(label) for label in labels]
        features = data.drop(self.predictor, axis = 1)
        features = np.array(features)
        return train_test_split(features, labels, test_size = 0.2)


if __name__ == "__main__":
    svm = SVM(RESULTS_DIR + 'oxo_game.csv',["Step","Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8","Agent"],["Move"])
    #svm = SVM(RESULTS_DIR + 'othello_game.csv',["Step","Cell_0_0","Cell_0_1","Cell_0_2","Cell_0_3","Cell_1_0","Cell_1_1","Cell_1_2","Cell_1_3","Cell_2_0","Cell_2_1","Cell_2_2","Cell_2_3","Cell_3_0","Cell_3_1","Cell_3_2","Cell_3_3","Agent"],["Move"])
    #svm = SVM(RESULTS_DIR + 'nim_game.csv',["Chips","Agent"],["Move"])
    svm.train()

    