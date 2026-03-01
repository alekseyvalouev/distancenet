 - We want to generate trajectories that are 4H units long. Then we will subsample like so:
 1   2   3   4   5
 o---o---o---o---o
 ^^^^^ 
 - We will then get all subtrajectories lengths H - 4H. For example:
    - 1-2, 2-3, 3-4, 4-5 (length H)
    - 1-3, 2-4, 3-5 (length 2H)
    - 1-4, 2-5 (length 3H)
    - 1-5 (length 4H)

 - For each of 1, 2, 3, 4, 5 let's also collect a negative sample. Take all scenes in the train set and pick a random node from each one to collect 5 negative samples. 
 - Make sure to execute the turn instruction on the start node but NOT on the end node. 
 - For each 4H length trajectory we will then classify them into bins of length 1, 2, 3, 4, -1. 
