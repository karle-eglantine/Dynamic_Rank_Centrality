# This script is written by E. Karlé to conduct the experiments of the article Dynamic Ranking with the BTL Model: A Nearest Neighbor based Rank Centrality Method https://arxiv.org/abs/2109.13743
#
# This script performs the analysis of synthetic data generated according to the Dynamic BTL model.
# The user can choose amongst the Dynamic Rank Centrality method, the Maximum Likelihood method or the Borda count method.
# It outputs the l_2, l_inf and error metric Dw of each estimators for all Monte Carlo runs and all values of T and N.

import random
import numpy as np
import pickle
import sys
import scipy.stats as ss

sys.path.append('modules')

import mle_module as mle
import graph_module as graph
import simulation_module as sim
import algorithms_module as algo
import time

def run_drc(list_N,list_T,L,B,c_delta,c1,cov_param,mu_param,rc_flag,mle_flag,borda_flag):
    '''
    This function run the DRC, MLE and Borda Count methods (up to the user choice) on synthetic data, and outputs their l2 and l_inf error, as well as the error on the ranks D_w(sigma). These results are automatically saved, along with the average running time and corresponding standard deviations for all the methods.
    
    Input :
    list_N: number of items/players
    list_T: number of seasons/different observations times
    L: number of comparisons done at time t if two items are compared
    output_grid: times for which we want to compute an estimator. It has to be included into np.arange(0,1,T) to compute the MSE
    B: number of bootstrap repetitions
    cov_param: covariance parameters to generate w*
    mu_param: mean parameters to generate w*
    c1,c2 : comparison graphs are generated as Erdös-Renyi with parameter p(t) drawn from the uniform distribution on [c1/N;c2/N]
    c_delta: constant involved in the choice of delta
    
    Output :
    results_RC : list containing the l2,Dw and linf errors for the DRC estimator.
    results_borda : list containing the Dw error for the Borda Count estimator.
    results_MLE : list containing the l2,Dw and linf errors for the MLE estimator.

    Note that each list output by this function contain elements of the form :
    MSE_RC: list  of lists of length len(list_N)xlen(list_T) such that MSE_RC[N][T] is a len(output_grid)-B array containing for each given time t and each estimation \pihat_B(t) the MSE = norm(\pihat_B(t)-w*(t))_2 for parameters N,T.
    error_RC : list  of lists of length len(list_N)xlen(list_T) such that error_RC[N][T] is a len(output_grid)-B array containing for each given time t and each estimation \pihat_B(t) the error D_\pistar(t)(\pihat_B(t)) for parameters N,T.
    l_inf_RC : list  of lists of length len(list_N)xlen(list_T) such that l_inf_RC[N][T] is a len(output_grid)-B array containing for each given time t and each estimation \pihat_B(t) the infinty norm = norm(\pihat_B(t)-w*(t))_infty for parameters N,T.
    '''
    
    random.seed(0)
    np.random.seed(0)
    
    # Initialize arrays that will gather the results
    # l2 error
    MSE_RC = [] # List of len(list_N) lists.
    MSE_MLE = []
    data_RC = []
    data_MLE = []
    # l_inf error
    l_inf_RC = []
    l_inf_MLE = []
    data_inf_RC = []
    data_inf_MLE = []
    # Dw(sigma) error on the ranks
    error_borda = []
    error_RC = []
    error_MLE = []
    
    # Runing time of methods
    time_rc = []
    time_mle = []
    time_borda = []
    sd_rc = []
    sd_mle = []
    sd_borda=[]
    
    
    # Analysis : loops for all values of T,N
    for N in list_N:
        c2 = np.log(N)
        MSE_RC_N = [] # List of len(list_T) arrays len(output_grid)xB, for a fixed N
        MSE_MLE_N = []
        data_RC_N = []
        data_MLE_N = []
        l_inf_RC_N = []
        l_inf_MLE_N = []
        data_inf_RC_N = []
        data_inf_MLE_N = []
        error_borda_N = []
        error_RC_N = []
        error_MLE_N = []
        
        time_rc_N = []
        time_mle_N = []
        time_borda_N = []
        sd_rc_N=[]
        sd_mle_N = []
        sd_borda_N = []
    
    
        
        for T in list_T: # adapt final arrays containing the results
            print(T)
            # Output grid
            #step_out = 1/(3*T)
            step_out = 1/T
            output_grid = np.arange(0,1+step_out/2,step_out)
            N_out = len(output_grid)
            grid = np.arange(0,T+1)
            
            
            # Initizialise intermediate arrays for results
            pi_RC = np.zeros((N_out,B,N))
            MSE_RC_TN = np.zeros((N_out,B))
            l_inf_RC_TN = np.zeros((N_out,B))
            error_RC_TN = np.zeros((N_out,B))
            
            pi_MLE = np.zeros((N_out,B,N))
            MSE_MLE_TN = np.zeros((N_out,B))
            l_inf_MLE_TN = np.zeros((N_out,B))
            error_MLE_TN = np.zeros((N_out,B))
            
            borda_TN = np.zeros((N_out,B,N))
            error_borda_TN = np.zeros((N_out,B))
            
            time_rc_TN = np.zeros(B)
            time_mle_TN = np.zeros(B)
            time_borda_TN = np.zeros(B)
            
            # Bootstrap loop
            for b in range(B):
                # Compute w*
                w = sim.w_gaussian_process(N, T+1 , mu_param, cov_param)
                w = w[grid,:]
                
                # Choose delta as theoretical optimal value
                delta = max(1/2,c_delta*T**(2/3))
                
                # Call get_valid_data
                vec_delta = delta*np.ones(N_out)
                A,vec_delta = sim.get_valid_graphs(N,T,vec_delta,output_grid,c1,c2)
                            
                
                # Generate pairwise information : Yl = L comparisons for each pair,Y = mean over the L comparisons
                Yl = sim.get_comparison_data(N,T,L,w)
                Y = A*np.mean(Yl,axis=1)
                
                # RC method
                if rc_flag:
                    start_rc = time.time()
                    for i,t in enumerate(output_grid):
                        pi_RC[i,b,:] = algo.RC_dyn(t,Y,A,vec_delta[i],tol = 1e-12) # Estimation
                        MSE_RC_TN[i,b] = np.linalg.norm(pi_RC[i,b,:]-w[i,:]/sum(w[i,:])) # MSE
                        l_inf_RC_TN[i,b] = np.linalg.norm(pi_RC[i,b,:]-w[i,:]/sum(w[i,:]),np.inf) # L inf error
                        error_RC_TN[i,b] = algo.error_metric(w[i,:]/sum(w[i,:]),ss.rankdata(-pi_RC[i,b,:],method='average')) # Dw error
                    end_rc = time.time()
                    time_rc_TN[b] = end_rc-start_rc
                    
                # Borda
                if borda_flag:
                    start_borda = time.time()
                    for i,t in enumerate(output_grid):
                        borda_TN[i,b,:] = algo.borda_count(t,Y,A,vec_delta[i])# Estimation
                        error_borda_TN[i,b] = algo.error_metric(w[i,:]/sum(w[i,:]),ss.rankdata(-borda_TN[i,b,:],method='average')) # Dw error
                    end_borda = time.time()
                    time_borda_TN[b] = end_borda-start_borda
                        
                
                # MLE method
                if mle_flag:
                    start_mle = time.time()
                    # Data for MLE
                    d_MLE = np.transpose(A*np.sum(Yl,axis=1),(0,2,1))
                    # MLE estimation
                    h = T**(-3/4) # bandwidth chosen w.r.t Bong's simulations.
                    T_list = np.arange(0,1+1/step_out)
                    ks_data = mle.kernel_smooth(d_MLE,h,T_list) # Smoothing of the data
                    pi_MLE[:,b,:] = np.exp(mle.gd_bt(data = ks_data,verbose=False)[1]) # Estimation
                    MSE_MLE_TN[:,b] = [np.linalg.norm(pi_MLE[i,b,:]/sum(pi_MLE[i,b,:])-w[i,:]/sum(w[i,:])) for i in range(N_out)] # MSE
                    l_inf_MLE_TN[:,b] = [np.linalg.norm(pi_MLE[i,b,:]/sum(pi_MLE[i,b,:])-w[i,:]/sum(w[i,:]),np.inf) for i in range(N_out)] # L_inf error
                    error_MLE_TN[:,b] = [algo.error_metric(w[i,:]/sum(w[i,:]),ss.rankdata(-pi_MLE[i,b,:],method='average')) for i in range(N_out)] # Dw error
                    
                    end_mle = time.time()
                    time_mle_TN[b] = end_mle-start_mle
                
            # Mean over bootstrap experiments
            if rc_flag:
                # MSE
                data_RC_N.append(np.ravel(MSE_RC_TN)) # Save all errors to be able to construct boxplot for the MSE
                MSE_RC_N.append(np.mean(MSE_RC_TN))
                # l_inf
                data_inf_RC_N.append(np.ravel(l_inf_RC_TN))
                l_inf_RC_N.append(np.mean(l_inf_RC_TN))
                # Dw
                error_RC_N.append(np.mean(error_RC_TN))
                # Running time
                time_rc_N.append(np.mean(time_rc_TN))
                sd_rc_N.append(np.std(time_rc_TN))
            if borda_flag:
                # Dw
                error_borda_N.append(np.mean(error_borda_TN))
                # Running time
                time_borda_N.append(np.mean(time_borda_TN))
                sd_borda_N.append(np.std(time_borda_TN))
            if mle:
                # MSE
                data_MLE_N.append(np.ravel(MSE_MLE_TN))
                MSE_MLE_N.append(np.mean(MSE_MLE_TN))
                # l_inf
                data_inf_MLE_N.append(np.ravel(l_inf_MLE_TN))
                l_inf_MLE_N.append(np.mean(l_inf_MLE_TN))
                # Dw
                error_MLE_N.append(np.mean(error_MLE_TN))
                # Running time
                time_mle_N.append(np.mean(time_mle_TN))
                sd_mle_N.append(np.std(time_mle_TN))
            
        if rc_flag:
            # MSE
            data_RC.append(data_RC_N)
            MSE_RC.append(MSE_RC_N)
            # l_inf
            data_inf_RC.append(data_inf_RC_N)
            l_inf_RC.append(l_inf_RC_N)
            # Dw
            error_RC.append(error_RC_N)
            # Running time
            time_rc.append(time_rc_N)
            sd_rc.append(sd_rc_N)
            # List of results
            results_RC = [MSE_RC,error_RC,l_inf_RC,data_RC,data_inf_RC,time_rc,sd_rc]
        
        if borda_flag:
            # Dw
            error_borda.append(error_borda_N)
            # Running time
            time_borda.append(time_borda_N)
            sd_borda.append(sd_borda_N)
            # List of results
            results_borda = [error_borda,time_borda,sd_borda]
        
        if mle_flag:
            # MSE
            data_MLE.append(data_MLE_N)
            MSE_MLE.append(MSE_MLE_N)
            # l_inf
            data_inf_MLE.append(data_inf_MLE_N)
            l_inf_MLE.append(l_inf_MLE_N)
            # Dw
            error_MLE.append(error_MLE_N)
            # Running time
            time_mle.append(time_mle_N)
            sd_mle.append(sd_mle_N)
            # List of results
            results_MLE = [MSE_MLE,error_MLE,l_inf_MLE,data_MLE,data_inf_MLE,time_mle,sd_mle]

    # Return the restults as lists.
    return [results_RC,results_borda,results_MLE]



