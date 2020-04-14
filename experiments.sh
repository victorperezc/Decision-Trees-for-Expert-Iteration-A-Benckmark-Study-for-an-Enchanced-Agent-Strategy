python3 main.py -a1 mcts -a2 mcts -g oxo -i 20 -n 10 -o mcts_mcts_oxo_continous.csv --continous
python3 main.py -a1 rfc -a2 mcts -g oxo -i 20 -n 10 -o rfc_mcts_oxo_continous.csv --continous
python3 main.py -a1 svm -a2 mcts -g oxo -i 20 -n 10 -o svm_mcts_oxo_continous.csv --continous
python3 main.py -a1 mlp -a2 mcts -g oxo -i 20 -n 10 -o mlp_mcts_oxo_continous.csv --continous

python3 main.py -a1 mcts -a2 mcts -g othello -i 20 -n 10 -o mcts_mcts_othello_continous.csv --continous
python3 main.py -a1 rfc -a2 mcts -g othello -i 20 -n 10 -o rfc_mcts_othello_continous.csv --continous
python3 main.py -a1 svm -a2 mcts -g othello -i 20 -n 10 -o svm_mcts_othello_continous.csv --continous
python3 main.py -a1 mlp -a2 mcts -g othello -i 20 -n 10 -o mlp_mcts_othello_continous.csv --continous

python3 main.py -a1 mcts -a2 mcts -g nim -i 20 -n 10 -o mcts_mcts_nim_continous.csv --continous
python3 main.py -a1 rfc -a2 mcts -g nim -i 20 -n 10 -o rfc_mcts_nim_continous.csv --continous
python3 main.py -a1 svm -a2 mcts -g nim -i 20 -n 10 -o svm_mcts_nim_continous.csv --continous
python3 main.py -a1 mlp -a2 mcts -g nim -i 20 -n 10 -o mlp_mcts_nim_continous.csv --continous

python3 main.py -a1 mcts -a2 mcts -g oxo -i 20 -n 10 -o mcts_mcts_oxo.csv 
python3 main.py -a1 rfc -a2 mcts -g oxo -i 20 -n 10 -o rfc_mcts_oxo.csv 
python3 main.py -a1 svm -a2 mcts -g oxo -i 20 -n 10 -o svm_mcts_oxo.csv 
python3 main.py -a1 mlp -a2 mcts -g oxo -i 20 -n 10 -o mlp_mcts_oxo.csv 

python3 main.py -a1 mcts -a2 mcts -g othello -i 20 -n 10 -o mcts_mcts_othello.csv 
python3 main.py -a1 rfc -a2 mcts -g othello -i 20 -n 10 -o rfc_mcts_othello.csv 
python3 main.py -a1 svm -a2 mcts -g othello -i 20 -n 10 -o svm_mcts_othello.csv 
python3 main.py -a1 mlp -a2 mcts -g othello -i 20 -n 10 -o mlp_mcts_othello.csv

python3 main.py -a1 mcts -a2 mcts -g nim -i 20 -n 10 -o mcts_mcts_nim.csv
python3 main.py -a1 rfc -a2 mcts -g nim -i 20 -n 10 -o rfc_mcts_nim.csv
python3 main.py -a1 svm -a2 mcts -g nim -i 20 -n 10 -o svm_mcts_nim.csv
python3 main.py -a1 mlp -a2 mcts -g nim -i 20 -n 10 -o mlp_mcts_nim.csv