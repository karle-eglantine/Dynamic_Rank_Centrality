# This module contains functions to perform Leave-One-Out Cross-Validation for the DRC,MLE and the Borda Count methods.
# The code for the MLE method belongs to Bong et al. (2020)

import sys
import numpy as np
import scipy as sc
import scipy.linalg as spl
import scipy.stats as ss

import mle_module as mle
import simulation_module as sim
   

def loocv_rc(Y,A,delta_list,num_loocv = 200,t=1):
    '''
    Y: T-N-N array containing the proportions of time i won against j at any time t
    A: T-N-N array containing the adjacency matrices of the graphs at each time t
    delta_list: list of candidates for the value of delta
    num_loocv : number of cross validation to perform
    t: time in [0,1] at which we want to estimate the ranks
    
    Return a optimal value of delta and the estimate of the weights at time t for this value of delta by the DRC method
    '''    
    T,N = np.shape(A)[:2]
    # Create a pool of estimates pi at time t to choose from
    pis = np.zeros((len(delta_list),N))
    for i,delta in enumerate(delta_list):
        pis[i,:] = sim.RC_dyn(t,Y,A, delta, tol=1e-12)
    
    
    indices = np.transpose(np.nonzero(A)) # Array of all (t,i,j) possible combinations to choose from
    N_comp = np.shape(indices)[0] # total number of comparisons
    
    error = np.zeros(len(delta_list)) # Array of the mean error of the loocv for each value of delta
    for l,delta in enumerate(delta_list):
        error_delta = np.zeros(num_loocv) # Array of the errors for each loocv for a given value of delta
        for k in range(num_loocv):
            # Copy the data
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            # Choose one comparison y_{ij}(t) to remove from the dataset
            rand_match = np.random.randint(N_comp) # random number between 0 and Total number of comparisons
            rand_index = indices[rand_match,:] # Select the tuple (t,i,j) corresponding to the rand_match comparison
            s,i,j = tuple(rand_index)
        
            # Remove the test value from the data
            Y_loocv[s,i,j] = max(Y_loocv[s,i,j]-1,0) # if all observations at these time where 0, then Y[t,i,j] stays 0.
            Y_loocv[s,j,i] = max(Y_loocv[s,j,i]-1,0)
            A_loocv[s,i,j] = 0
            A_loocv[s,j,i] = 0  
            
            # Fit model and compute prediction error
            pi = sim.RC_dyn(s/T,Y_loocv,A_loocv, delta, tol=1e-12) # vector of length N
            prob = pi[j]/(pi[i]+pi[j])
            error_delta[k] = np.linalg.norm(prob-Y[s,i,j])
        # Compute the mean error for each value of delta
        error[l] = np.mean(error_delta)
         
    # Choose the value of delta that minimizes the error, and the corresponding estimate pi(t)
    index = max(idx for idx, val in enumerate(error) if val == np.min(error[~np.isnan(error)]))
    delta_star = delta_list[index]
    pi_star = pis[index,:]
    
    return delta_star,pi_star

def loocv_borda(Y,A,delta_list,t,num_loocv = 200):
    '''
    Y: T-N-N array containing the proportions of time i won against j at any time t
    A: T-N-N array containing the adjacency matrices of the graphs at each time t
    delta_list: list of candidates for the value of delta
    num_loocv : number of cross validation to perform
    t: time in [0,1] at which we want to estimate the ranks
    
    Return a optimal value of delta and the estimate of the weights at time t for this value of delta by the Borda Count method
    '''    
    T,N = np.shape(A)[:2]
    # Create a pool of estimates pi at time t to choose from, using the Borda Count method
    pis = np.zeros((len(delta_list),N))
    for i,delta in enumerate(delta_list):
        pis[i,:] = sim.borda_count(t,Y,A, delta)
    
    
    indices = np.transpose(np.nonzero(A)) # Array of all (t,i,j) possible combinations to choose from
    N_comp = np.shape(indices)[0] # total number of comparisons
    
    error = np.zeros(len(delta_list))
    for l, delta in enumerate(delta_list):
        error_delta = np.zeros(num_loocv)
        for k in range(num_loocv):
            # Copy the data
            Y_loocv = Y.copy()
            A_loocv = A.copy()
            
            # Choose one comparison to remove from the dataset
            rand_match = np.random.randint(N_comp) # random number between 0 and Total number of comparisons
            rand_index = indices[rand_match,:] # Select the tuple (t,i,j) corresponding to the rand_match comparison
            s,i,j = tuple(rand_index)
        
            # Remove the test value from the data
            Y_loocv[s,i,j] = max(Y_loocv[s,i,j]-1,0) # if all observations at these time where 0, then Y[t,i,j] stays 0.
            Y_loocv[s,j,i] = max(Y_loocv[s,j,i]-1,0)
            A_loocv[s,i,j] = 0
            A_loocv[s,j,i] = 0  
            
            # Fit the model and compute prediction error
            pi = sim.borda_count(s/T,Y_loocv,A_loocv, delta) # vector of length N
            ranks = ss.rankdata(-pi,method='average')
            error_delta[k] = (Y[s,i,j]==1)*(ranks[i]< ranks[j])
            
        error[l] = np.mean(error_delta)
    
    # Choose the value of delta that minimizes the error and the corresponding estimate 
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
    
