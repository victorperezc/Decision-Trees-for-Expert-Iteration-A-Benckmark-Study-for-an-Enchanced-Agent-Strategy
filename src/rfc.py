import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pickle
from src.definitions import CHARTS_DIR,REGRESSORS_DIR,RESULTS_DIR

class RandomForest():

    def __init__(self,dataset,features,predictor,classes,name,cm=False,vi=False,dump_regressor=False):
        self.data = dataset
        self.features = features
        self.predictor = predictor
        self.classes = classes
        self.name = name
        self.cm = cm
        self.vi = vi
        self.dump_regressor = dump_regressor
        self.rf = RandomForestClassifier(n_estimators = 50, bootstrap = True, max_features = 'sqrt', random_state = 34,n_jobs=-1)

    # Train the Random Forest Classifier
    def train(self):
        train_features, test_features, train_labels, test_labels = self.split(self.data)
        self.rf.fit(train_features, train_labels)

        train_predictions, train_probs = self.test(self.rf,train_features,train_labels)
        test_predictions, test_probs = self.test(self.rf,test_features,test_labels)

        if self.cm:
            cm = confusion_matrix(test_labels, test_predictions)
            self.plot_confusion_matrix(cm, self.classes, title = self.name + " Confusion Matrix")

        if self.vi:
            self.featureImportances()

        if self.dump_regressor:
            with open(self.dump_regressor, 'wb') as fid:
                pickle.dump(self.rf, fid)  

        return self.accuracy(self.rf,test_features,test_labels)


    def test(self,model,features,labels):
        predictions = model.predict(features)
        probs = model.predict_proba(features)[:, 1]
        #print("Accuracy : ", self.accuracy(model,features,labels))
        return predictions,probs

    def predict(self,d):
        return self.rf.predict(d)[0]

    def accuracy(self,model,predictions,labels):
        return model.score(predictions, labels)

    def featureImportances(self):
        importances = list(self.rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(self.features, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        plt.figure(figsize = (10, 10))
        plt.style.use('fivethirtyeight')
        x_values = list(range(len(importances)))
        plt.bar(x_values, importances, orientation = 'vertical')
        plt.xticks(x_values, self.features, rotation='45',fontsize=7)
        plt.ylabel('Importance')
        plt.xlabel('Variable')
        plt.title(self.name + ' Variable Importances')
        plt.savefig(self.vi)


        return importances

    def split(self,data):
        labels = np.array(data[self.predictor])
        labels = [int(label) for label in labels]
        features = data.drop(self.predictor, axis = 1)
        features = np.array(features)
        return train_test_split(features, labels, test_size = 0.2)

        
    def plot_confusion_matrix(self,cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Oranges):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        # Plot the confusion matrix
        plt.figure(figsize = (10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size = 24)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size = 14)
        plt.yticks(tick_marks, classes, size = 14)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        
        # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
            
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size = 18)
        plt.xlabel('Predicted label', size = 18)

    # Confusion matrix

        plt.savefig(self.cm)


