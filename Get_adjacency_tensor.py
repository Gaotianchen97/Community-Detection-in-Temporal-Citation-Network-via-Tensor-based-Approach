# -*- coding: utf-8 -*-
"""
Paper: Community Detection in Temporal Citation Network via Tensor-based Approach

Author: Tianchen Gao
"""

#Load the required packages
import numpy as np                           # Basic 
import networkx as nx                        # Network
import warnings                              # Basic
warnings.filterwarnings("ignore")


# Function: Constructing adjacency matrix
def get_graph(df,Year1,Year2):
    '''
    input: 
        df: edges data (dataframe)
        Year1: the year of the edges; 
        Year2: the year of the nodes.
    output: A: an adjacency matrix.
    '''
    # Construct the network in year2
    df_list_1 = df[(df['Year'] >= 2001) & (df['Year'] <= Year2)][['edge2','edge1']] 
    edge_list = [] 
    for i in range(len(df_list_1['edge1'])):                                  
        cite_cited = df_list_1.iloc[i,1],+ df_list_1.iloc[i,0]                  
        edge_list.append(cite_cited)     
    G  = nx.DiGraph()                                                     
    G.add_edges_from(edge_list)  
    # Get the set of considered nodes                                
    nodes_list = G.nodes()
    # Construct the network in year1
    df_list_2 = df[(df['Year'] >= 2001) & (df['Year'] <= Year1)][['edge2','edge1']] 
    edge_list = []                                                                  
    for i in range(len(df_list_2['edge1'])): 
        if (df_list_2.iloc[i,1] in nodes_list) &  (df_list_2.iloc[i,0] in nodes_list):  # Select the edges
            cite_cited = df_list_2.iloc[i,1],+ df_list_2.iloc[i,0]                      
            edge_list.append(cite_cited)                                           
    G = nx.DiGraph()                                                              
    G.add_nodes_from(nodes_list)                                    
    G.add_edges_from(edge_list)                                      
    A = nx.to_numpy_matrix(G).getA()   # Get adjacency matrix in year1
    return A                           # Return A

def get_tensor(df,nodes_list,year,t):
    
    '''
    input: 
        df: edges data (dataframe)
        nodes_list: the list of nodes;
        year: the last year; 
        t: the time span.
    output: T: an adjacency tensor.
    
    '''
    T = np.zeros((len(nodes_list),t,len(nodes_list)))           # Create zero tensor
    for i in range(t):
        T[:,i,:] = np.asmatrix(get_graph(df,year-t+i+1,year))   # Assignment
    
    return T






