# CE888-Project

## Student

Victor Perez Cester ( vp19885@essex.ac.uk )

## Set Up Dependencies

You must execute the project under a Linux environment.

Install the requirements as 

    pip3 install -r requirements.txt

## Execute 

Run from the command line as

    python3 main.py 

The system includes a wide range of command line arguments to customize the execution.

    -h, --help            show the help message

    -a1 {mcts,rfc,svm,mlp}, --agent1 {mcts,rfc,svm,mlp} Agent 1 Training Method

    -a2 {mcts,rfc,svm,mlp}, --agent2 {mcts,rfc,svm,mlp} Agent 2 Training Method

    -o OUTPUT, --output OUTPUT Output Filename

    -i ITERATIONS, --iterations ITERATIONS Number of Iterations

    -c, --continous       If the training should be carried out using continous data

    -n NGAMES, --ngames NGAMES Number of games to play per iteration

    -g {oxo,othello,nim}, --game {oxo,othello,nim} Game to play

    -v, --verbose         Program Verbosity

## Default Execution
When executing withou argument these are the default arguments used

    python3 main.py -a1 rfc -a2 rfc -g oxo -i 10 -n 10 -o date_now.csv

## Execution examples

Execute the code with both agents playing with MCTS

    python3 main.py --agent1 mcts --agent2 mcts


Execute with Agent 1 playing with RFC and agent 2 with MCTS

    python3 main.py -a1 rfc -a2 mcts -g oxo -i 100 -n 20 -o outputfile.csv --continous --verbose