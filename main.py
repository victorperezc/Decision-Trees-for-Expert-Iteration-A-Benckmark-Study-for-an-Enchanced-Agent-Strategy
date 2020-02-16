import sys, getopt
from src.uct import OXOState,OthelloState,NimState,UCT
from src.dataCollector import DataCollector
from src.rfc import RandomForest
import argparse
import pandas as pd
import datetime
from src.definitions import ROOT_DIR,RESULTS_DIR

class ParseArguments():

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("-a1","--agent1",help="Agent 1 Training Method", choices=["mcts","rfc","svm","mlp"],default="rfc")
        parser.add_argument("-a2","--agent2",help="Agent 2 Training Method", choices=["mcts","rfc","svm","mlp"],default="rfc")
        parser.add_argument("-o","--output", help="Output Filename",default=datetime.datetime.now().strftime("%d.%m.%Y%_H:%M:%S.csv"))
        parser.add_argument("-i", "--iterations", help="Number of Iterations",default=10,type=int)
        parser.add_argument("-c","--continous",action="store_true",help="If the training should be carried out using continous data",default=False)
        parser.add_argument("-n", "--ngames", help="Number of games to play per iteration",default=10,type=int)
        parser.add_argument("-g","--game", help="Game to play",choices=["oxo","othello","nim"],default="oxo")
        parser.add_argument("-v","--verbose",action="store_true", help="Program Verbosity",default=False)

        return parser.parse_args()

class Main():

    def __init__(self):
        self.args = ParseArguments.parse()
        self.collector = DataCollector(args=self.args)
        self.classifiers = {
            "mcts" : "MCTS",
            "rfc" : "RFC",
            "svm" : "SVM",
            "mlp" : "MLP"
            }
        self.results = []
        
    def train(self):
        if self.args.verbose: print("[INFO] Simulating {0} iterations and {1} games per iteration for the {2} game determining Agent 1 moves using a {3} and Agent 2 moves using a {4}.".format(self.args.iterations,self.args.ngames,self.args.game,self.classifiers[self.args.agent1],self.classifiers[self.args.agent2]))
        classifier = None
        continous_df = None
        record = []
        for i in range(self.args.iterations):
            if classifier:
                df,wins = self.collector.collect(iteration=i,games=self.args.ngames,classifier=classifier)
            else:
                df,wins = self.collector.collect(iteration=i,games=self.args.ngames)

            if self.args.continous:
                if continous_df is None:
                    continous_df = df
                else:
                    continous_df = continous_df.append(df)

            if self.args.agent1 != "mcts" or self.args.agent2 != "mcts":

                classifier = RandomForest(continous_df if self.args.continous else df,
                    features=["Step","Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8","Agent"],
                    predictor=["Move"],
                    classes=["Cell_0","Cell_1", "Cell_2","Cell_3","Cell_4","Cell_5","Cell_6","Cell_7","Cell_8"],
                    name="OXO Game"
                )

                record.append(classifier)

                self.results.append({
                    "Iteration" : i,
                    "Accuracy" : round(classifier.train(),3),
                    "Player 1 Wins" : wins["1"],
                    "Player 2 Wins" : wins["2"],
                    "Nobody Wins" : wins["0"]
                })
            else:

                self.results.append({
                    "Iteration" : i,
                    "Accuracy" : "0",
                    "Player 1 Wins" : wins["1"],
                    "Player 2 Wins" : wins["2"],
                    "Nobody Wins" : wins["0"]
                })


        df = pd.DataFrame(self.results)
        if self.args.verbose: print("[INFO] Writing results to {0}".format(RESULTS_DIR + self.args.output))
        df.to_csv(RESULTS_DIR + self.args.output, index=False)


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