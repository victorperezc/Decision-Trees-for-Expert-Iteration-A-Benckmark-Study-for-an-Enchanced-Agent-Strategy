
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/victor/Desktop/Erasmus/CE888-Data-Mining/CE888-Project/src/data/results/svm_mcts_othello.csv',usecols=["Player 1 Wins","Player 2 Wins","Nobody Wins"])
p1wins = df['Player 1 Wins'].sum()
p2wins = df["Player 2 Wins"].sum()
ties = df["Nobody Wins"].sum()
total = p1wins + p2wins + ties
print("Player 1 win ratio ", p1wins/total)
print("Player 2 win ratio ", p2wins/total)
print("Ties ratio ", ties/total)
print("Player 1 vs Player 2 win ratio ",p1wins/p2wins)
df.plot.bar()
plt.xlabel('# Iteration')
plt.ylabel('# Games')
plt.savefig('output.png')