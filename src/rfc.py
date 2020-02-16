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
from definitions import SIMULATIONS_DIR,CHARTS_DIR,REGRESSORS_DIR

class RandomForest():

    def __init__(self,filename,features,predictor,classes,name,cm=False,vi=False,dump_regressor=False):
        self.filename = filename
        self.features = features
        self.predictor = predictor
        self.classes = classes
        self.name = name
        self.cm = cm
        self.vi = vi
        self.dump_regressor = dump_regressor
        self.rf = RandomForestClassifier(n_estimators = 50, bootstrap = True, max_features = 'sqrt', random_state = 34,n_jobs=-1)

    def ramdomHyperparametersSearch(self):
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter=10, cv = 10, verbose=2, n_jobs = -1)
        data = self.loadData(self.features,self.predictor)
        train_features, test_features, train_labels, test_labels = self.split(data)
        rf_random.fit(train_features, train_labels)
        print(rf_random.best_params_)
        self.test(rf_random,test_features,test_labels)


    # Train the Random Forest Classifier
    def train(self):
        data = self.loadData(self.features,self.predictor)
        train_features, test_features, train_labels, test_labels = self.split(data)
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


    def test(self,model,features,labels):
        predictions = model.predict(features)
        probs = model.predict_proba(features)[:, 1]
        print("Accuracy : ", self.accuracy(model,features,labels))
        return predictions,probs

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

    def loadData(self,features,predictor):
        data = pd.read_csv(self.filename,usecols=features + predictor)
        return data

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



if __name__ == "__main__":

    rf = RandomForest(filename=SIMULATIONS_DIR + 'oxo_game.csv',
        features=["Step","Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8","Agent"],
        predictor=["Move"],
        classes=["Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8"],
        name="OXO Game",
        cm=CHARTS_DIR + "oxo_game_confusion_matrix.png",
        vi=CHARTS_DIR + "oxo_game_variable_importances.png",
        dump_regressor=REGRESSORS_DIR + "oxo_game_dumped_rfc.pkl"
    ).train()
    """
    rf = RandomForest(filename=SIMULATIONS_DIR + 'othello_game.csv',
        features=["Step","Cell_0_0","Cell_0_1","Cell_0_2","Cell_0_3","Cell_1_0","Cell_1_1","Cell_1_2","Cell_1_3","Cell_2_0","Cell_2_1","Cell_2_2","Cell_2_3","Cell_3_0","Cell_3_1","Cell_3_2","Cell_3_3","Agent"],
        predictor=["Move"],
        classes=["Cell_0_0","Cell_0_1","Cell_0_2","Cell_0_3","Cell_1_0","Cell_1_3","Cell_2_0","Cell_2_3","Cell_3_0","Cell_3_1","Cell_3_2","Cell_3_3"],
        name="OXO Game",
        cm=CHARTS_DIR + "othello_game_confusion_matrix.png",
        vi=CHARTS_DIR + "othello_game_variable_importances.png",
        dump_regressor=REGRESSORS_DIR + "othello_game_dumped_rfc.pkl"
    ).train()

    rf = RandomForest(filename=SIMULATIONS_DIR + 'nim_game.csv',
        features=["Step","Chips","Agent"],
        predictor=["Move"],
        classes=["Chips_1","Chips_2","Chips_3"],
        name="OXO Game",
        cm=CHARTS_DIR + "nim_game_confusion_matrix.png",
        vi=CHARTS_DIR + "nim_game_variable_importances.png",
        dump_regressor=REGRESSORS_DIR + "nim_game_dumped_rfc.pkl"
    ).train()
    """