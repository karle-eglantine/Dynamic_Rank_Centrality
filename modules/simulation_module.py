import scipy as sc
from scipy import stats
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

sys.path.append('Documents/GitHub/Dynamic_Rank_Centrality/synthetic_data')

import graph_module as graph

def borda_count(t,Y,A,delta):
    T, N = A.shape[:2]
    N_delta = graph.neighborhood(t,delta,T-1)
    Y_delta = np.sum(Y[N_delta,:,:],axis=0)
    A_delta = np.sum(A[N_delta,:,:],axis=0)
    borda_count = np.sum(Y_delta, axis=0) / np.sum(A_delta,axis=0)
    return borda_count

def error_metric(w,sigma):
    s = 0
    N = np.shape(w)[0]
    for i in range(N):
        for j in range(i+1,N):
            cond = (w[i]-w[j])*(sigma[i]-sigma[j])>0
            if cond:
                s =+ (w[i]-w[j])**2 # Only count pairs where ranking is incorrect
    return np.sqrt(s/(2*N*np.linalg.norm(w)**2))

def rank_list(v):
    output = [0] * len(v)
    for i, x in enumerate(sorted(range(len(v)), key=lambda y: v[y])):
        output[x] = i
    return np.array(output)

def av_dif_rank(w1,w2):
    # both w should be T-by-N matrices
    T, N = w1.shape
    result = [0] * T
    for t in range(T):
        result[t] = np.mean(abs(rank_list(w1[t]) - rank_list(w2[t])))
    return result

def w_gaussian_process(N, T, mu_parameters, cov_parameters, mu_type = 'constant', cov_type = 'toeplitz'):
    '''
    generate w of shape TxN via a Gaussian process : w = exp(beta) where beta is generated via a Gaussian process
    '''
    if mu_type == 'constant':
        loc, scale = mu_parameters
        mu_start = stats.norm.rvs(loc = loc,scale = scale,size = N,random_state = 100)
        mu = [np.ones(T) * mu_start[i] for i in range(N)]
    if cov_type == 'toeplitz':
        alpha, r = cov_parameters
    ##### strong auto-correlation case, off diagonal  = 1 - T^(-alpha) * |i - j|^r
    off_diag = 1 - np.float_power(T,-alpha) * np.float_power(np.arange(1,T + 1),r)
    cov_single_path = sc.linalg.toeplitz(off_diag,off_diag)

    return np.exp(np.array([np.random.multivariate_normal(mean = mu[i],cov = cov_single_path,size = 1).ravel() for i in range(N)]).T)

# or generate,N functions of time, indep of T
def generate_w(N,T):
    '''return a TxN vector with w_i(t) = 2+2*(i-1)/n + cos(2*pi*t)'''
    w = np.zeros((T,N))
    for i in range(N):
        for t in range(T):
            w[t,i] = 2+2*i/np.sqrt(N) + np.cos(2*np.pi*t/T)
    return w

def get_adjacency_graphs(N,T,c1 = 1,c2 = 10):
    '''
    get a list of T+1 adjacency matrices of G(n,p) graphs on the inputgrid
    -------------
    Output :
    A: (T+1)-N-N array where A[t,:,:] is the adjacency matrix of a G(n,p(t)) at time t. p(t) is drawned uniformly from [c_1/n,c_2/n]
    '''
    A = np.zeros((T+1,N,N))
    for t in range(T+1):
        p = np.random.uniform(c1/N,c2/N)
        for i in range(N):
            for j in range(i+1,N):
                A[t,i,j] = np.random.binomial(1,p)
                A[t,j,i] = A[t,i,j]
    return A

def get_comparison_data(N,T,L,w):
    '''
    get (T+1)*L comparison matrices
    -------------
    Output :
Yl: (T+1)-L-N-N array where Yl[t,l,:,:] is a matrix of pairwise information y_ij^l(t), following Bernoulli distribution of parameter p_ij(t) = w_tj/(w_ti+w_tj)
    '''
    Yl = np.zeros((T+1,L,N,N))
    for t in range(T+1):
        for l in range(L):
            for i in range(N):
                for j in range(i+1,N):
                    Yl[t,l,i,j] = np.random.binomial(1,w[t,j]/(w[t,j]+w[t,i]))
                    Yl[t,l,j,i] = 1-Yl[t,l,i,j]
    return Yl


def get_valid_graphs(N,T,vec_delta,output_grid,c1,c2):
    '''
    Generate list of T+1 adjacency matrices and vector of values of delta such that for any t in the output_grid, G_delta(t) is connected
    ---------------------
    Input :
    vec_delta : vector of same length as output_grid, initialized to be the constant vector with value delta^* = c_delta*T**(2/3)/(N*L**1/3)
    output_grid : list of times at which we want to recover the ranks
    c1,c2 : constant for the choice of the probabilities of the Erdos Renyi input graphs : p(t) is Uniform([c1/N;c2/N]) 
    
    Output:
    A : list of valid adjacency matrices
    vec_delta : updated values of delta, for wich G_delta(t) is connected (t in output_grid)
    '''
    A = get_adjacency_graphs(N,T,c1,c2)
    # Check if union of all the graphs is connected
    A_delta = graph.union_graph(A,T,0.5)
    while not graph.connected(A_delta):
        A = get_adjacency_graphs(N,T,c1,c2)
        A_delta = graph.union_graph(A,T,0.5)
    # Now we are sure that we will be able to find valid values for delta.
    # Compute the union_graph
    grid = np.arange(0,1+1/(2*T),1/T)
    for i,t in enumerate(output_grid):
        A_delta = graph.union_graph(A,vec_delta[i],t)
        N_delta = graph.neighborhood(t,vec_delta[i],T)
        while sum(N_delta) == 0 or not graph.connected(A_delta):
            # Add graphs on each side of the neighborhood
            idx = graph.update_neighborhood(N_delta) # List of indexes of points of the grid in the neighborhood. We add (if possible) one grid point on the left and on the right to the old neighborhood.
            vec_delta[i] = T*max(t-grid[idx[0]],grid[idx[-1]]-t)
            A_delta = graph.union_graph(A,vec_delta[i],t)
    return A,vec_delta


print('simu')


### Spectral module ####
import scipy.linalg as spl

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



