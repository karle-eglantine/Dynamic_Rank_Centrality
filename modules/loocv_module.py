# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking with the BTL Model: A Nearest Neighbor based Rank Centrality Method https://arxiv.org/abs/2109.13743
#
# This script contains all the tools to perform Cross Validation for DRC and MLE methods, in order to tune their parameters, respectively delta and the bandwidth h.
# The function loocv_mle is extracted from Shamindra Shrotriya's GitHub repository, as a part of their experiments in their paper : Bong, Heejong and Li, Wanshan and Shrotriya, Shamindra and Rinaldo, Alessandro. "Nonparametric Estimation in the Dynamic Bradley-Terry Model" The 23rd International Conference on Artificial Intelligence and Statistics. 2020.

# Copyright (c) 2020 Shamindra Shrotriya


import sys
import numpy as np
import scipy as sc
import scipy.linalg as spl
import scipy.stats as ss

import mle_module as mle
import simulation_module as sim
   

def loocv_rc(Y,A,delta_list,num_loocv = 200,t=1):
    '''
    Performs Cross-validation for tuning the parameter delta in the DRC method.
    Input:
        Y: T-N-N array of observations
        A: T-N-N array of adjacency matrices
        delta_list: list of parameters delta to choose from
        num_loocv: number of cross validation iterations.
        t : time at which we want to recover the ranks (between 0 and 1)
    Output:
        delta_star : chosen value of delta
        pi_star : estimation of pi(t) by the DRC method using delta = delta_star 
    '''    
    T,N = np.shape(A)[:2]
    # Create a pool of pi to choose from
    pis = np.zeros((len(delta_list),N))
    for i,delta in enumerate(delta_list):
        pis[i,:] = algo.RC_dyn(t,Y,A, delta, tol=1e-12)
    
    
    indices = np.transpose(np.nonzero(A)) # Array of all (t,i,j) possible combinations to choose from
    N_comp = np.shape(indices)[0] # total number of comparisons
    
    error = np.zeros(len(delta_list))
    for l,delta in enumerate(delta_list):
        error_delta = np.zeros(num_loocv)
        for k in range(num_loocv):
            Y_loocv = Y.copy()
            A_loocv = A.copy()
        
            rand_match = np.random.randint(N_comp) # random number between 0 and Total number of comparisons
            rand_index = indices[rand_match,:] # Select the tuple (t,i,j) corresponding to the rand_match comparison
            s,i,j = tuple(rand_index)
        
            # Remove the test value from the data
            Y_loocv[s,i,j] = max(Y_loocv[s,i,j]-1,0) # if all observations at these time where 0, then Y[t,i,j] stays 0.
            Y_loocv[s,j,i] = max(Y_loocv[s,j,i]-1,0)
            A_loocv[s,i,j] = 0
            A_loocv[s,j,i] = 0  
            
            # Fit model and compute prediction error
            pi = algo.RC_dyn(s/T,Y_loocv,A_loocv, delta, tol=1e-12) # vector of length N
            prob = pi[j]/(pi[i]+pi[j])
            error_delta[k] = np.linalg.norm(prob-Y[s,i,j])
        error[l] = np.mean(error_delta)

    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    delta_star = delta_list[index]
    pi_star = pis[index,:]
    
    return delta_star,pi_star


def loocv_mle(data, h_list, opt_fn,
          num_loocv = 200, get_estimate = True, return_prob = True,
          verbose = 'cv', out = 'terminal', **kwargs):
    '''
    conduct local
    ----------
    Input:
    data: TxNxN array
    h: a vector of kernel parameters
    opt_fn: a python function in a particular form of 
        opt_fn(data, lambda_smooth, beta_init=None, **kwargs)
        kwargs might contain hyperparameters 
        (e.g., step size, max iteration, etc.) for
        the optimization function
    num_loocv: the number of random samples left-one-out cv sample
    get_estimate: whether or not we calculate estimates beta's for 
        every lambdas_smooth. If True, we use those estimates as 
        initial values for optimizations with cv data
    verbose: controlling the verbose level. If 'cv', the function 
        prints only cv related message. If 'all', the function prints
        all messages including ones from optimization process.
        The default is 'cv'.
    out: controlling the direction of output. If 'terminal', the function
        prints into the terminal. If 'notebook', the function prints into 
        the ipython notebook. If 'file', the function prints into a log 
        file 'cv_log.txt' at the same directory. You can give a custom 
        output stream to this argument. The default is 'terminal'
    **kwargs: keyword arguments for opt_fn
    ----------
    Output:
    lambda_cv: lambda_smooth chosen after cross-validation
    nll_cv: average cross-validated negative loglikelihood 
    beta_cv: beta chosen after cross-validation. None if get_estimate is False
    '''    
    h_list = h_list.flatten()
    h_list = -np.sort(-h_list)
    betas = [None] * h_list.shape[0]
    
    last_beta = np.zeros(data.shape[:2])
    for i, h in enumerate(h_list):
        
        ks_data = mle.kernel_smooth(data,h)
        _, beta = opt_fn(ks_data, beta_init = last_beta, **kwargs)
        betas[i] = beta.reshape(data.shape[:2])
        last_beta = betas[i]
        
    indices = np.array(np.where(np.full(data.shape, True))).T
    cum_match = np.cumsum(data.flatten())
    
    if out == 'terminal':
        out = sys.__stdout__
    elif out == 'notebook':
        out = sys.stdout
    elif out == 'file':
        out = open('cv_log.txt', 'w')
    
    loglikes_loocv = np.zeros(h_list.shape)
    prob_loocv = np.zeros(h_list.shape)
    for i in range(num_loocv):
        data_loocv = data.copy()
        rand_match = np.random.randint(np.sum(data))
        rand_index = indices[np.min(np.where(cum_match >= rand_match)[0])]
        data_loocv[tuple(rand_index)] -= 1

        for j, h in enumerate(h_list):
            ks_data_loocv = mle.kernel_smooth(data_loocv,h)
            _, beta_loocv = opt_fn(ks_data_loocv, beta_init=betas[j],
                                   verbose=(verbose in ['all']), out=out, **kwargs)
            beta_loocv = beta_loocv.reshape(data.shape[:2])
            loglikes_loocv[j] += beta_loocv[rand_index[0],rand_index[1]] \
                   - np.log(np.exp(beta_loocv[rand_index[0],rand_index[1]])
                            + np.exp(beta_loocv[rand_index[0],rand_index[2]]))
            prob_loocv[j] += 1 - np.exp(beta_loocv[rand_index[0],rand_index[1]]) \
                / (np.exp(beta_loocv[rand_index[0],rand_index[1]])
                    + np.exp(beta_loocv[rand_index[0],rand_index[2]]))

        if verbose in ['cv', 'all']:
            out.write("%d-th cv done\n"%(i+1))
            out.flush()
    if return_prob:
        return (h_list[np.argmax(loglikes_loocv)], -loglikes_loocv[::-1]/num_loocv, 
                betas[np.argmax(loglikes_loocv)], prob_loocv[::-1]/num_loocv)
    else:
        return (h_list[np.argmax(loglikes_loocv)], -loglikes_loocv[::-1]/num_loocv, 
                betas[np.argmax(loglikes_loocv)])
    
