import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from definitions import SIMULATIONS_DIR

class SVM():

    def __init__(self,filename,features,predictor):
        self.filename = filename
        self.features = features
        self.predictor = predictor
        self.classifier = SVC(kernel='rbf')

    # Train the Random Forest Classifier
    def train(self):
        data = self.loadData(self.features,self.predictor)
        train_features, test_features, train_labels, test_labels = self.split(data)
        self.classifier.fit(train_features, train_labels)
        y_pred = self.classifier.predict(test_features)
        print(confusion_matrix(test_labels,y_pred))
        print(classification_report(test_labels,y_pred))

        with open('my_dumped_classifier.pkl', 'wb') as fid:
            pickle.dump(self.classifier, fid)  

    def test(self,model,features,labels):
        predictions = model.predict(features)
        probs = model.predict_proba(features)[:, 1]
        print("Accuracy : ", self.accuracy(model,features,labels))
        return predictions,probs
        
    def accuracy(self,model,predictions,labels):
        return model.score(predictions, labels)

    def loadData(self,features,predictor):
        data = pd.read_csv(self.filename,usecols=features + predictor)
        return data

    def split(self,data):
        labels = np.array(data[self.predictor])
        labels = [int(label) for label in labels]
        features = data.drop(self.predictor, axis = 1)
        features = np.array(features)
        return train_test_split(features, labels, test_size = 0.2)


if __name__ == "__main__":
    svm = SVM(SIMULATIONS_DIR + 'oxo_game.csv',["Step","Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8","Agent"],["Move"])
    #svm = SVM(SIMULATIONS_DIR + 'othello_game.csv',["Step","Cell_0_0","Cell_0_1","Cell_0_2","Cell_0_3","Cell_1_0","Cell_1_1","Cell_1_2","Cell_1_3","Cell_2_0","Cell_2_1","Cell_2_2","Cell_2_3","Cell_3_0","Cell_3_1","Cell_3_2","Cell_3_3","Agent"],["Move"])
    #svm = SVM(SIMULATIONS_DIR + 'nim_game.csv',["Chips","Agent"],["Move"])
    svm.train()

    