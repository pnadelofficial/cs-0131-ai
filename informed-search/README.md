# Assignment 02 - Informed Search

## Usage
    * Make sure your working directory is the "informed-search" directory with the `pwd` command.

    * Once you are in that directory, simply run the file with `python pancake_problem.py`. This will solve the pancake problem on a random ordering of 1 through 10 using the gap heuristic. 

    * You can also specify a list that you want the script to solve by writing the list after the script command. For example, `python pancake_problem.py 3 2 5 1 4` would solve the pancake problem for the list [3, 2, 5, 1, 4].
    
    * There are also a number of flags which alter behavior, as will be expanded upon below. 
        - `-u` or `--ucs` will solve the pancake problem with the Uniform Cost Search algorithm. 
        
        - `-t` or `--top` will solve the pancake problem with A*, using the top heuristic. 
        
        - `-p` or `--topprime` will solve the pancake problem with A*, using the top' heuristic. 
        
        - And `-l` or `--ltopprime` will solve the pancake problem with A*, using the L-top' heuristic.  

    * At the conclusion of the script, you should see reporting on what steps the algorithm took to solve the problem, how long it took, and how many nodes were visited.

## Assumptions
My solution assumes that the input sequence wil be made up of consecutive integers, despite them not being in consecutive order. 