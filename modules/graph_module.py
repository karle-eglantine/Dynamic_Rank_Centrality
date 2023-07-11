# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking with the BTL Model: A Nearest Neighbor based Rank Centrality Method https://arxiv.org/abs/2109.13743
#
# This module contains functions used to construct neighborhoods, graphs and check some graph properties.

import numpy as np
import networkx as nx

def neighborhood(t,delta,T):
    '''
    get an array of length T+1 with boolean True if grid[i] belongs to the neighborhood of t
    '''
    grid = np.arange(0,1+1/(2*T),1/T)
    return abs(grid-t)<= delta/T

def graph_proba(graph_matrix):
    '''
    Estimate the probaility of a G(n,p)
    '''
    N = np.shape(graph_matrix)[0]
    return 2*np.sum(np.triu(graph_matrix,k=1))/(N*(N-1))

def union_graph(A,delta,t):
    '''
    get the adjacency matrix of the union graph G_delta(t), each coefficient is equal to the number of graphs in the neighborhood involved in the edge (i,j).
    A: (T+1)-N-N array containing the adjacency matrix of the observed graphs.
    '''
    T,N = np.shape(A)[0:2]
    N_delta = neighborhood(t,delta,T-1)
    A_delta = sum(A[N_delta,:,:])
    return A_delta

def connected(A):
    ''' Check if a graph is connected from its adjacency matrix'''
    G = nx.from_numpy_matrix(A)
    return nx.is_connected(G)

def update_neighborhood(neigh):
    ''' Extend a neighborhood from one grid point on the left, and one on the right (if possible)
    Input :
        neigh : list of length nb_gridpoints, fillled with booleans indicating if a grid point is in the neighborhood or not.'''
    T = np.shape(neigh)[0]
    idx = np.nonzero(neigh)[0]
    if np.shape(idx)[0] == T: # The neighborhood is already equal to the complete grid
        return idx
    else:
        new_neigh = neigh
        if idx[0] == 0: # Can't complete the neighborhood on the left : already include time t = 0
            new_neigh[idx[-1]+1] = True # Add one point on the right
        else:
            new_neigh[idx[0]-1] = True # Add a point on the left
            if idx[-1]!= T-1: # Can also add a point on the right
                new_neigh[idx[-1]+1] = True
    return np.nonzero(new_neigh)[0]

def complete(A):
    ''' Check if a graph is complete from its adjacency matrix'''
    N = np.shape(A)[0]
    G = nx.from_numpy_matrix(A)
    return nx.graph_clique_number(G) == N

def list_edges(M):
    '''
    Get the list of pairs of indexes (i,j), i<j, such that (i,j) is an edge in the graph
    Input :
        M : adjacency matrix of a graph
    Output :
        list of pairs
    '''
    non_zero = np.transpose(np.nonzero(np.triu(M)))
    return list(non_zero)

print('graph')