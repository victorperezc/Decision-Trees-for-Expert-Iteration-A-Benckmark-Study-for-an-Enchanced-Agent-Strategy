
from sklearn.neural_network import MLPClassifier as MLPClass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from definitions import CHARTS_DIR,REGRESSORS_DIR,RESULTS_DIR

class MLPClassifier():

    def __init__(self,filename,features,predictor,classes,name,cm=False,vi=False,dump_regressor=False,solver='adam', alpha=1e-5, hidden_layer_sizes=(100), random_state=1):
        self.filename = filename
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
        data = self.loadData(self.filename,self.features,self.predictor)
        train_features, test_features, train_labels, test_labels = self.split(data)
        self.classifier.fit(train_features, train_labels)

        self.test(test_features,test_labels)

    def test(self,features,labels):
        predictions = self.classifier.predict(features)
        print("ACCURACY : ",self.accuracy(labels,predictions))
        self.confusionMatrix(labels,predictions)

    def accuracy(self,labels,predictions):
        return accuracy_score(labels,predictions)

    def confusionMatrix(self,labels,predictions):
        print(confusion_matrix(labels, predictions))

    def loadData(self,file,features,predictor):
        data = pd.read_csv(file,usecols=features + predictor)
        return data

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

if __name__ == "__main__":
    rf = MLPClassifier(filename=RESULTS_DIR + 'oxo_game.csv',
        features=["Step","Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8","Agent"],
        predictor=["Move"],
        classes=["Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8"],
        name="OXO Game",
        cm=CHARTS_DIR + "oxo_game_ann_confusion_matrix.png",
        vi=CHARTS_DIR + "oxo_game_ann_variable_importances.png",
        dump_regressor=REGRESSORS_DIR + "oxo_game_dumped_rfc.pkl"
    ).train()