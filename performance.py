ƒ# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking with the BTL Model: A Nearest Neighbor based Rank Centrality Method https://arxiv.org/abs/2109.13743
#
# This script computes the MSE of the DRC estimator, for different value of delta. This will show how the choice of the window impacts the estimation

import random
import numpy as np
import pickle
import sys
import scipy.stats as ss

sys.path.append('modules')

import mle_module as mle
import graph_module as graph
import simulation_module as sim


def perf_delta(N,T,list_delta,L,B,c1,cov_param,mu_param):
    '''
        Parameters :
        N: number of items/players
        T: number of seasons/different observations times
        list_delta : list of window parameters delta to test
        L: number of comparisons done at time t if two items are compared
        output_grid: times for which we want to compute an estimator. It has to be included into np.arange(0,1,T) to compute the MSE
        B: number of bootstrap repetitions
        cov_param: covariance parameters to generate w*
        mu_param: mean parameters to generate w*
        c1,c2 : comparison graphs are generated as Erdös-Renyi with parameter p(t) drawn from the uniform distribution on [c1/N;c2/N]
        
        Output :
        MSE_RC: list of all the mean MSE (computed for B Monte Carlo runs) for different values of window parameter delta, given N and T.
        '''
    
    random.seed(0)
    np.random.seed(0)
    
    # Initialize arrays that will gather the results
    # l2 error
    MSE_RC = np.zeros(len(list_delta)) # List of len(list_delta) lists.
    
    
    # Analysis : loops for all values of delta
    c2 = np.log(N)
    # Output grid
    #step_out = 1/(3*T)
    step_out = 1/T
    output_grid = np.arange(0,1+step_out/2,step_out)
    N_out = len(output_grid)
    grid = np.arange(0,T+1)
            
    for d,delta in enumerate(list_delta):
        print('delta='+str(delta))
        # Initizialise intermediate arrays for results
        pi_RC = np.zeros((N_out,B,N))
        MSE_RC_delta = np.zeros((N_out,B))
            
        # Bootstrap loop
        for b in range(B):
            # Compute w*
            w = sim.w_gaussian_process(N, T+1 , mu_param, cov_param)
            w = w[grid,:]
    
            # Simulation of observation graphs
            vec_delta = delta*np.ones(N_out)
            A,vec_delta = sim.get_valid_graphs(N,T,vec_delta,output_grid,c1,c2)
    
    
            # Generate pairwise information : Yl = L comparisons for each pair,Y = mean over the L comparisons
            Yl = sim.get_comparison_data(N,T,L,w)
            Y = A*np.mean(Yl,axis=1)
    
            # DRC method
            for i,t in enumerate(output_grid):
                pi_RC[i,b,:] = algo.RC_dyn(t,Y,A,vec_delta[i],tol = 1e-12)
                MSE_RC_delta[i,b] = np.linalg.norm(pi_RC[i,b,:]-w[i,:]/sum(w[i,:]))
    
        # Mean over all bootstrap experiments and all timepoints for this value of delta
        MSE_RC[d] = np.mean(MSE_RC_delta)

    return MSE_RC
    
    
    
