# CE888-Project

## Student

Victor Perez Cester ( vp19885@essex.ac.uk )

## Set Up Dependencies

Please note that this project has been developed under a MacOS distribution. I highly recommned you to execute it under a MacOS or Unix distribution. It's not been tested under a windows environment.

Create a virtual environment 

    python3 -m venv env
    
Install the requirements as 

    pip3 install -r requirements.txt

## Usage

Run from the command line as

    python3 main.py 
    
The usage is the following

    usage: main.py [-h] [-a1 {mcts,rfc,svm,mlp}] [-a2 {mcts,rfc,svm,mlp}]
                   [-o OUTPUT] [-i ITERATIONS] [-c] [-n NGAMES]
                   [-g {oxo,othello,nim}] [-v] [-nc NIMCHIPS]
                   [-ob OTHELLOBOARDSIZE]

The system includes a wide range of command line arguments to customize the execution.

    -h, --help. Show this help message and exit
    -a1 {mcts,rfc,svm,mlp}, --agent1 {mcts,rfc,svm,mlp}. Agent 1 Training Method
    -a2 {mcts,rfc,svm,mlp}, --agent2 {mcts,rfc,svm,mlp}. Agent 2 Training Method
    -o OUTPUT, --output OUTPUT. Output Filename
    -i ITERATIONS, --iterations ITERATIONS. Number of Iterations
    -c, --continous. If the training should be carried out using continous data
    -n NGAMES, --ngames NGAMES. Number of games to play per iteration
    -g {oxo,othello,nim}, --game {oxo,othello,nim}. Game to play
    -v, --verbose. Program Verbosity
    -nc NIMCHIPS, --nimChips NIMCHIPS. Nim Chips
    -ob OTHELLOBOARDSIZE, --othelloBoardSize OTHELLOBOARDSIZE. Othello Board Size

## Default Execution
When executing without arguments these are the default arguments used

    python3 main.py -a1 rfc -a2 rfc -g oxo -i 10 -n 10 -o date_now.csv

## Execution examples

Execute the code with both agents playing with MCTS

    python3 main.py --agent1 mcts --agent2 mcts


Execute with Agent 1 playing with RFC and agent 2 with MCTS

    python3 main.py -a1 rfc -a2 mcts -g oxo -i 100 -n 20 -o outputfile.csv --verbose

## Continous training

The agents can be trained using continous data pipeline, that means that the agent n is trained using all the outcomes from the previous agents (0,n-1). Therefore, the agents not only are trained on the past agent data but on all the generations. To achieve so you can add the flag --continous to your execution. By default continous learning is disabled and therefore an agent n will be trained with only the last agent output, also kown as 1-interval window training.

    python3 main.py -a1 rfc -a2 mcts -g oxo -i 100 -n 20 -o outpufile.csv --continous --verbose

    
