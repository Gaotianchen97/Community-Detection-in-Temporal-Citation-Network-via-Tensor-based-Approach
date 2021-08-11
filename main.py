# -*- coding: utf-8 -*-
"""
Paper: Community Detection in Temporal Citation Network via Tensor-based Approach

Author: Tianchen Gao

"""


"""
0. Preparatory Work
"""
#Load the required packages
import pandas as pd                          # Basic
import networkx as nx                        # Network
import warnings                              # Basic
warnings.filterwarnings("ignore")
from CP_decomposition import decompose_three_way   
from TD_SCORE import TD_SCORE
from Get_adjacency_tensor import get_tensor
"""
1. Real Data Analysis
"""

# Step 1: Construct the network
# Read data
df = pd.read_csv("D:/资料/论文/动态网络社区发现/代码整理/paper_core.csv")   # Read data of core subgraph
# Construct edges
edge_list = []                                    # Create list
for i in range(len(df['edge1'])):                 # Traverse nodes
    cite_cited = df.iloc[i,1],+ df.iloc[i,0]      # edge1 cite edge2
    edge_list.append(cite_cited)                  # Generate an edge format that can be used to construct a directed network
# Construct nodes
nodes_list = list(df['edge1'])+list(df['edge2'])  # Connect edges
nodes_list = list(set(nodes_list))                # Deduplication
# Construct a directed graph
G = nx.DiGraph()                                  # Construct an empty graph
G.add_edges_from(edge_list)                       # Construct edges from the set of connected edges (not self-connected)
print("The number of nodes:",len(G.nodes()))      # Print the number of nodes
print("The number of edges:",len(G.edges()))      # Print the number of edges
print("The density of network:",nx.density(G))    # Print the density of the netowrk
nodes_list_2018 = list(G.nodes)


#Step 2: Construct adjacency tensor
T18 = get_tensor(df,nodes_list_2018,2018,18)      # df: the edges data, nodes_list_2018: the list of nodes in the last year, 2018: the last year, 18: the time span.
print(T18.shape)                                  # Print the shape of T18

# Step 3: CP decomposition
# Parameters: 
rank = 7                                          # The number of components
# CP decomposition
U,t,V = decompose_three_way(T18, 7)               # CP decomposition

# Step 4: TD-SCORE
# Parameters: 
set_K_comm = 7                                    # The number of communities
# Labels
labels = TD_SCORE(T18, rank, set_K_comm, U,t,V)   # Labels
print(labels)                                     # Print labels



