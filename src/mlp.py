
from sklearn.neural_network import MLPClassifier as MLPClass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.definitions import CHARTS_DIR,REGRESSORS_DIR,RESULTS_DIR

class MLPClassifier():

    def __init__(self,dataset,features,predictor,classes,name,cm=False,vi=False,dump_regressor=False,solver='adam', alpha=1e-5, hidden_layer_sizes=(100), random_state=1):
        self.data = dataset
        self.features = features
        self.predictor = predictor
        self.classes = classes
        self.name = name
        self.cm = cm
        self.vi = vi
        self.dump_regressor = dump_regressor
        self.classifier = MLPClass(
            solver=solver, 
            alpha=alpha, 
            hidden_layer_sizes=hidden_layer_sizes, 
            random_state=random_state
        )
        self.scaler = StandardScaler()

    def train(self):
        train_features, test_features, train_labels, test_labels = self.split(self.data)
        self.classifier.fit(train_features, train_labels)
        return self.accuracy(test_features,test_labels)

    def test(self,features,labels):
        predictions = self.classifier.predict(features)
        self.confusionMatrix(labels,predictions)

    def accuracy(self,labels,predictions):
        return self.classifier.score(labels,predictions)

    def confusionMatrix(self,labels,predictions):
        print(confusion_matrix(labels, predictions))

    def predict(self,d):
        return self.classifier.predict(d)[0]


    def split(self,data):
        labels = np.array(data[self.predictor])
        labels = [int(label) for label in labels]
        features = data.drop(self.predictor, axis = 1)
        features = np.array(features)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.15)
        self.scaler.fit(train_features)
        train_features = self.scaler.transform(train_features)
        test_features = self.scaler.transform(test_features)
        return train_features, test_features, train_labels, test_labels
