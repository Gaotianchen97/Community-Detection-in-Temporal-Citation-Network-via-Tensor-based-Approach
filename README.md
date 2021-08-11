# Community-Detection-in-Temporal-Citation-Network-via-Tensor-based-Approach
This package contains the data and the computer code to reproduce the results in Gao, Pan, Zhang and Wang's paper titled "Community Detection in Temporal Citation Network via Tensor-based Approach".

## Data
It contains 1 data file:
paper_core.csv -- the core subgraph of a temporal citation network with 4101 nodes and 48409 edges. 

## Code 
The Code folder contains our own code to reproduce the results in the paper, as well as the Python code for CP decomposition with ALS (Kolda and Bader, 2009), TD-SCORE and get adjacency tensor:

main.py -- the main code to reproduce the results in the paper

CP_decomposition.py -- the functions used in the CP decomposition with ALS

TD_SCORE.py -- the functions used in the TD_SCORE

Get_adjacency_tensor.py -- the functions used to get adjacency tensor

## references
Kolda T G, Bader B W. Tensor decompositions and applications[J]. SIAM review, 2009, 51(3): 455-500.
