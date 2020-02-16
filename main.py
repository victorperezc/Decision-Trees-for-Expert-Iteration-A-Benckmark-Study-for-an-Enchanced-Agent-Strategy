import sys, getopt
from src.uct import OXOState,OthelloState,NimState,UCT
from src.dataCollector import DataCollector
from src.rfc2 import RandomForest
import argparse
import pandas as pd
from src.definitions import ROOT_DIR,RESULTS_DIR

class ParseArguments():

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--classifier", help="Classifier Type", choices=["rfc","svm","mlp"],required=True)
        parser.add_argument("-o","--output", help="Output Filename",required=True)
        parser.add_argument("-i", "--iterations", help="Number of Iterations",default=10,type=int)
        parser.add_argument("-n", "--ngames", help="Number of games to play per iteration",default=10,type=int)
        parser.add_argument("-g","--game", help="Game to play",choices=["oxo","othello","nim"],required=True)
        parser.add_argument("-v","--verbose",action="store_true", help="Program Verbosity",default=False)

        return parser.parse_args()

class Main():

    def __init__(self):
        self.args = ParseArguments.parse()
        self.collector = DataCollector(args=self.args)
        self.classifiers = {"rfc" : "Random Forest Classifier"}
        self.results = []
        
    def train(self):
        if self.args.verbose: print("Training a {0} for {1} iterations and {2} games per iteration".format(self.classifiers[self.args.classifier], self.args.iterations,self.args.ngames))
        classifier = None
        record = []
        for i in range(self.args.iterations):
            if classifier:
                df,wins = self.collector.collect(iteration=i,games=self.args.ngames,classifier=classifier)
            else:
                df,wins = self.collector.collect(iteration=i,games=self.args.ngames)

            classifier = RandomForest(df,
                features=["Step","Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8","Agent"],
                predictor=["Move"],
                classes=["Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8"],
                name="OXO Game"
            )

            self.results.append({
                "Iteration" : i,
                "Accuracy" : round(classifier.train(),3),
                "Player 1 Wins" : wins["1"],
                "Player 2 Wins" : wins["2"],
                "Nobody Wins" : wins["0"]
            })

            record.append(classifier)

        df = pd.DataFrame(self.results)
        if self.args.verbose: print("Writing results to {0}".format(RESULTS_DIR + self.args.output))
        df.to_csv(RESULTS_DIR + self.args.output, index=False)
        _,wins_1 = self.collector.collect(iteration=0,games=self.args.ngames,classifier=record[0])
        _,wins_2 = self.collector.collect(iteration=0,games=self.args.ngames,classifier=record[-1])
        print(wins_1,wins_2)


"""
Arguments : 
    * -h or --help : Displays help information
    * -c or --classifier : [rfc,svm,mlp] | Stands for Random Forest Classifier, Support Vector Machine and Multi Layer Perceptron
    * -i or --iterations : Retraining iterations
    * -n or --ngames : # Games Per iteration 
    * -g or --game : Game to play [oxo,othello,nim]
"""
if __name__ == "__main__":
    Main().train()