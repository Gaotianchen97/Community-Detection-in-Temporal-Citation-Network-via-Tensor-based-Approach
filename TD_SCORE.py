# -*- coding: utf-8 -*-
"""
Paper: Community Detection in Temporal Citation Network via Tensor-based Approach

Author: Tianchen Gao
"""

#Load the required packages
import numpy as np                           # Basic 
from math import log                         # Basic
from sklearn.cluster import KMeans           # K-means
from copy import deepcopy                    # Deepcopy
import warnings                              # Basic
warnings.filterwarnings("ignore")



# Algorithm 2: Tensor Directed-Spectral Clustering on Ratios of Eigenvectors

def TD_SCORE(A,K_num, K_comm,U,t,V):
    '''
    input: 
        A: adjacency tensor; 
        K_num: the number of compontents; 
        K_comm: the number of communities; 
        U, t, V: the result of CP decomposition.
    
    output: Community label of each node
    '''
    
    u1 = U[:,0]                     # The first column vector of the matrix U
    v1 = V[:,0]                     # The first column vector of the matrix V
    
    #Setting parameters             
    n = len(A)                      # The number of nodes
    T = len(t)                      # The adjacency tensor is n*n*T dimension
    T_n = log(n)                    # Calculate log(n), n>3
   
    
    # Generate N1 and N2 according to u1 and v1, where N1 is the set of nodes corresponding to the non-zero value in u1, and N2 is the set of nodes corresponding to v1
    ob_supp_u1 = [i for i,a in enumerate(u1) if a==0]    # The set of nodes corresponding to the value of 0 in u1
    ob_supp_v1 = [i for i,a in enumerate(v1) if a==0]    # The set of nodes corresponding to the value of 0 in v1
    N1 = list(set(range(n)).difference(set(ob_supp_u1))) # Get N1 
    N2 = list(set(range(n)).difference(set(ob_supp_v1))) # Get N2
    
    # Calculate R_star_l (n*k-1)                       
    R_star_l = np.zeros((len(A), K_num-1))               # Generate n*k-1 dimensional matrix
    for i in range(K_num-1):                             # Traverse all nodes
        R_star_l[:,i] = U[:,i+1]/u1                      # Divide all vectors by u1
    R_star_l[R_star_l > T_n] = T_n                       # If the absolute value of R_star_l is greater than log(n), then log(n)
    R_star_l[R_star_l < -T_n] = -T_n                          
    for i in ob_supp_u1:                                 # Traverse nodes not in N1
        R_star_l[i,:] = np.zeros(K_num-1)                # Assign a value of 0 to nodes not in N1
        
    # CalculateR_star_r (n*k-1)
    R_star_r = np.zeros((len(A), K_num-1))               # Generate n*k-1 dimensional matrix
    for i in range(K_num-1):                             # Traverse all nodes
        R_star_r[:,i] = V[:,i+1]/v1                      # Divide all vectors by v1
    R_star_r[R_star_r > T_n] = T_n                       # If the absolute value of R_star_l is greater than log(n), then log(n)
    R_star_r[R_star_r < -T_n] = -T_n
    for i in ob_supp_v1:                                 # Traverse nodes not in N2
        R_star_r[i,:] = np.zeros(K_num-1)                # Assign a value of 0 to nodes not in N2
    
    #N1+N2
    #Restrict the rows of R(l) and R(r) to the set N1+N2
    N1_N2 = list(set(N1).intersection(set(N2)))          # N1_N2 is the intersection of N1 and N2
    R_res_l = R_star_l[N1_N2,:]                          # Corresponding to N1_N2 in R_star_l
    R_res_r = R_star_r[N1_N2,:]                          # Corresponding to N1_N2 in R_star_r
    R_res = np.hstack((R_res_l, R_res_r))                # Stack arrays horizontally
    N1_N2_kmeans = KMeans(n_clusters = K_comm,n_init=20,  max_iter=1000, init = 'k-means++').fit(R_res) # Kmeans clustering on N1+N2
        
    # Construct a dictionary to record the results of the community
    community_dict = {}                                  # Construct a dictionary
    community_label = list(set(N1_N2_kmeans.labels_))    # Record the Kmeans clustering result of the intersection of N1 and N2. Classification label [0,K_num-1]
    dict_attribute = {'center_l':[], 'center_r':[], 'cluster_l':[], 'cluster_r':[], 'index':[]}   # Initialize the dictionary
    community_dict = community_dict.fromkeys(community_label)   # Construct a dictionary
    for i in community_dict.keys():                             
        community_dict[i] = deepcopy(dict_attribute)            # deepcopy
    # Assignment
    # Find each community cluster center
    cluster_R_res_l = list(zip(R_res_l, N1_N2_kmeans.labels_, N1_N2)) # Pack R_res_l, N1_N2_kmeans.labels_ (clustering labels), N1_N2 (node) into a list
    cluster_R_res_r = list(zip(R_res_r, N1_N2_kmeans.labels_, N1_N2)) # Pack R_res_r, N1_N2_kmeans.labels_ (clustering labels), N1_N2 (node) into a list
    for item in cluster_R_res_l:                                      
        community_dict[item[1]]['cluster_l'].append(item[0])         
        community_dict[item[1]]['index'].append(item[2])             
    for item in cluster_R_res_r:                                      
        community_dict[item[1]]['cluster_r'].append(item[0])          
    for i in community_dict.keys():                                   
        cluster_l = np.array(community_dict[i]['cluster_l'])                 
        community_dict[i]['center_l'] = np.sum(cluster_l, axis = 0)/len(community_dict[i]['cluster_l'])  # Right cluster center of class k
    for i in community_dict.keys():
        cluster_r = np.array(community_dict[i]['cluster_r'])                                      
        community_dict[i]['center_r'] = np.sum(cluster_r, axis = 0)/len(community_dict[i]['cluster_r'])  # Left cluster center of class k
    
    # N1/N2
    # The set of nodes in N1 that are not in N2
    N1eN2 = list(set(N1).difference(set(N1_N2)))    # Extract nodes that are in N1 but not in N2
    #A matrix including all of the central vector in different communities
    center_matrix = []                              # Define a matrix containing the left center vectors of different communities
    for value in community_dict.values():           
        center_matrix.append(value['center_l'])     # Record the left center vector of each community in the clustering result
    center_matrix = np.array(center_matrix)         
    # Measure the distance between certain node and center vector of each community by Euclidean Distance 
    R_res_l_N1eN2 = R_star_l[N1eN2,:]               # The matrix R_star_l corresponding to the node in N1 but not in N2
    for item in zip(R_res_l_N1eN2, N1eN2):          # Pack R_res_l_N1eN2 and N1eN2 (node) into a list
        distance = np.apply_along_axis(lambda a: np.sqrt(np.sum(a**2)),1,(center_matrix - item[0]))     # Calculate the distance to each cluster center vector
        one_label = distance.argmin()               # Nearest label
        community_dict[one_label]['index'].append(item[1]) # Add the node to the index of one_label of community_dict
    
    # N2/N1
    # The set of nodes in N2 that are not in N1
    N2eN1 = list(set(N2).difference(set(N1_N2)))    # Extract nodes that are in N12 but not in N1
    #A matrix including all of the central vector in different communities
    center_matrix = []                              # Define a matrix containing the right center vectors of different communities
    for value in community_dict.values():           
        center_matrix.append(value['center_r'])     # Record the right center vector of each community in the clustering result
    center_matrix = np.array(center_matrix)         
    # Measure the distance between certain node and center vector of each community by Euclidean Distance 
    R_res_r_N2eN1 = R_star_r[N2eN1,:]               # The matrix R_star_r corresponding to the node in N2 but not in N1
    for item in zip(R_res_r_N2eN1, N2eN1):          # Pack R_res_r_N1eN2 and N2eN1 (node) into a list
        distance = np.apply_along_axis(lambda a: np.sqrt(np.sum(a**2)),1,(center_matrix - item[0]))     # Calculate the distance to each cluster center vector
        one_label = distance.argmin()               # Nearest label
        community_dict[one_label]['index'].append(item[1]) # Add the node to the index of one_label of community_dict
        
    # /N1/N2
    # The set of nodes that are neither in N1 nor in N2
    eN2eN1 = list(set(range(n)).difference(set(list(set(N1).union(set(N2))))))                  # Extract nodes that are neither in N2 nor in N1
    R_res_eN2eN1_out= A[eN2eN1,T-1,:]                                                           # The part of the tensor out (out-degree) of the above node
    R_res_eN2eN1_in = A[:,T-1,eN2eN1]                                                           # The part of the tensor in (in-degree) of the above node
    for item in zip(R_res_eN2eN1_out, R_res_eN2eN1_in.T,eN2eN1):                                        
        connect_node = list(set(np.r_[np.argwhere(item[0] == 1),np.argwhere(item[1] == 1)][:,0]))# The collection of nodes connected to this node
        common_node_num = []                                                                    # The number of edges
        for i in community_dict.keys():                                                         
            common_node_num.append(len(set(community_dict[i]['index']) & set(connect_node)))    # Calculate the number of nodes connected to it
        one_label = np.array(common_node_num).argmax()                                          # Record the label with the most edges
        community_dict[one_label]['index'].append(item[2])                                      # Add the node to the community
        
    # Result
    label_vec = []                                                               
    for i in community_dict.keys():                                              
        n = len(community_dict[i]['index'])                                      # Record the number of nodes of communities
        label_vec.extend(list(zip([i]*n, community_dict[i]['index'])))           
    order_label_vec = sorted(label_vec, key = lambda label_vec: label_vec[1] )   # Sort by index
    label = np.array([item[0] for item in order_label_vec])                      # Label
    n_1 = len(N1_N2)                                                             # Print the number of nodes in N1_N2
    n_2 = len(N1eN2)                                                             # Print the number of nodes in N1eN2
    n_3 = len(N2eN1)                                                             # Print the number of nodes in N2eN1
    n_4 = len(eN2eN1)                                                            # Print the number of nodes in eN2eN1
    print(n_1,n_2,n_3,n_4)
    return label         