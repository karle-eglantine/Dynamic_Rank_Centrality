# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking with the BTL Model: A Nearest Neighbor based Rank Centrality Method https://arxiv.org/abs/2109.13743
#
# This module contains the main DRC algorithm, the Borda Count algorithm as well as the computation of the error metric Dw(sigma).

import scipy as sc
from scipy import stats
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import scipy.linalg as spl

sys.path.append('modules')

import graph_module as graph


def transition_matrix(A_delta,Y,N_delta):
    '''
    Compute the transition matrix of the union graph at time t
    Input : 
        A_delta : adajcency matrix of the union graph
        Y : (T+1)-N-N array containig the pairwise comparison information, averaged on the L workers at each time
        N_delta : neighborhood
    Output : N-N arary
    '''
    N = np.shape(Y)[1]
    transition_matrix = np.zeros((N,N))
    d = 2*N*graph.graph_proba(A_delta)
    for i in range(N):
        for j in range(N):
            if (i != j) & (A_delta[i,j] != 0):
                transition_matrix[i,j] = sum(Y[N_delta,i,j])/(d*A_delta[i,j])
        transition_matrix[i,i] = 1-np.sum(transition_matrix[i,:])
    return transition_matrix

def RC_dyn(t,Y,A,delta,tol = 1e-12):
    '''
    get the estimator pihat_RC(t)
    ------------
    Input :
    Y: (T+1)-N-N array containig the pairwise comparison information, averaged on the L workers at each time
    A: (T+1)-N-N array, A[t,:,:] is the adjacency matrix of data at time t
    tol = tolerance to approximate the eigenvalues equal to 1. 
    Output :
    If union graph disconnected
    '''
    T,N = np.shape(A)[:2]
    N_delta = graph.neighborhood(t,delta,T-1)
    A_delta = graph.union_graph(A,delta,t)
    P = transition_matrix(A_delta,Y,N_delta)
    eigval,eigvec = spl.eig(P,left=True,right=False)
    pi_RC = eigvec[:,abs(eigval - 1) < tol][:,0]
    return pi_RC/sum(pi_RC)


def borda_count(t,Y,A,delta):
    ''' Computes the Borda Count score of each item at time t
    Input
        t : estimation time, between 0 and 1
        Y: T-N-N array of observations
        A: T-N-N array of adjacency matrices
        delta: window parameter to compute the union graph
    '''
    T, N = A.shape[:2]
    N_delta = graph.neighborhood(t,delta,T-1)
    Y_delta = np.sum(Y[N_delta,:,:],axis=0)
    A_delta = np.sum(A[N_delta,:,:],axis=0)
    borda_count = np.sum(Y_delta, axis=0) / np.sum(A_delta,axis=0)
    return borda_count

def error_metric(w,sigma):
    ''' Computes the error metric D_w(sigma) for a given ranking sigma and given weights w'''
    s = 0
    N = np.shape(w)[0]
    for i in range(N):
        for j in range(i+1,N):
            cond = (w[i]-w[j])*(sigma[i]-sigma[j])>0
            if cond:
                s =+ (w[i]-w[j])**2 # Only count pairs where ranking is incorrect
    return np.sqrt(s/(2*N*np.linalg.norm(w)**2))