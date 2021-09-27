import random
import numpy as np
import pickle
import sys
import scipy.stats as ss

# Import files from folder 'modules'
import mle_module as mle
import graph_module as graph
import simulation_module as sim



'''
    Parameters :
    N: number of items/players
    T: number of seasons/different observations times
    L: number of comparisons done at time t if two items are compared
    output_grid: times for which we want to compute an estimator. It has to be included into np.arange(0,1,T) to compute the MSE
    B: number of bootstrap repetitions
    cov_param: covariance parameters to generate w*
    mu_param: mean parameters to generate w*
    c1,c2 : comparison graphs are generated as Erdös-Renyi with parameter p(t) drawn from the uniform distribution on [c1/N;c2/N]
    c_delta: constant involved in the choice of delta
    
    Output :
    MSE_RC: list  of lists of length len(list_N)xlen(list_T) such that MSE_RC[N][T] is a len(output_grid)-B array containing for each given time t and each estimation \pihat_B(t) the MSE = norm(\pihat_B(t)-w*(t))_2 for parameters N,T.
    MSE_MLE : same thing but for the MLE method.
    '''

random.seed(0)
np.random.seed(0)

# Parameters
list_N = [100]
list_T = np.arange(10,160,10)
L = 5
B = 20 # nb of Bootstrap repetitions
c_delta = 0.5 # constant for the choice of delta
c1 = 1 # Rather take a constant ex: c2  = o(log n/n) or constant
cov_param = [1,1]
mu_param = [0,0.1]

parameters = [list_N,list_T,L,B]

mle_flag=True
rc_flag=True
borda_flag=True


# Initialize arrays that will gather the results
MSE_RC = [] # List of len(list_N) lists.
MSE_MLE = []
error_borda = []
error_RC = []
error_MLE = []

# Analysis : loops for all values of T,N
for N in list_N:
    c2 = np.log(N)
    MSE_RC_N = [] # List of len(list_T) arrays len(output_grid)xB, for a fixed N
    MSE_MLE_N = []
    error_borda_N = []
    error_RC_N = []
    error_MLE_N = []
    
    for T in list_T: # adapt final arrays containing the results
        # Output grid
        #step_out = 1/(3*T)
        step_out = 1/T
        output_grid = np.arange(0,1+step_out/2,step_out)
        N_out = len(output_grid)
        grid = np.arange(0,T+1)
        
        
        # Initizialise intermediate arrays for results
        pi_RC = np.zeros((N_out,B,N))
        MSE_RC_TN = np.zeros((N_out,B))
        error_RC_TN = np.zeros((N_out,B))
        pi_MLE = np.zeros((N_out,B,N))
        MSE_MLE_TN = np.zeros((N_out,B))
        error_MLE_TN = np.zeros((N_out,B))
        borda_TN = np.zeros((N_out,B,N))
        error_borda_TN = np.zeros((N_out,B))
        
        # Bootstrap loop
        for b in range(B):
            # Compute w*
            w = sim.w_gaussian_process(N, T+1 , mu_param, cov_param)
            w = w[grid,:]
            
            # Choose delta
            #delta = max(1/2,c_delta*T**(2/3)/(L**(1/3)*N))
            # not a good choice : N too large
            delta = max(1/2,c_delta*T**(2/3))
            
            # Call get_valid_data
            vec_delta = delta*np.ones(N_out)
            A,vec_delta = sim.get_valid_graphs(N,T,vec_delta,output_grid,c1,c2)
                        
            
            # Generate pairwise information : Yl = L comparisons for each pair,Y = mean over the L comparisons
            Yl = sim.get_comparison_data(N,T,L,w)
            Y = A*np.mean(Yl,axis=1)
            
            # RC method
            if rc_flag:
                for i,t in enumerate(output_grid):
                    pi_RC[i,b,:] = sim.RC_dyn(t,Y,A,vec_delta[i],tol = 1e-12)
                    MSE_RC_TN[i,b] = np.linalg.norm(pi_RC[i,b,:]-w[i,:]/sum(w[i,:]))
                    error_RC_TN[i,b] = sim.error_metric(w[i,:]/sum(w[i,:]),ss.rankdata(-pi_RC[i,b,:],method='average'))
            
            # Borda
            if borda_flag:    
                for i,t in enumerate(output_grid):
                    borda_TN[i,b,:] = sim.borda_count(t,Y,A,vec_delta[i])
                    error_borda_TN[i,b] = sim.error_metric(w[i,:]/sum(w[i,:]),ss.rankdata(-borda_TN[i,b,:],method='average'))
            
            # MLE method
            if mle_flag:
                # Data for MLE
                data_MLE = np.transpose(A*np.sum(Yl,axis=1),(0,2,1))
                # MLE estimation
                h = T**(-3/4)
                T_list = np.arange(0,1+1/step_out)
                ks_data = mle.kernel_smooth(data_MLE,h,T_list)
                pi_MLE[:,b,:] = np.exp(mle.gd_bt(data = ks_data,verbose=True)[1])
                MSE_MLE_TN[:,b] = [np.linalg.norm(pi_MLE[i,b,:]/sum(pi_MLE[i,b,:])-w[i,:]/sum(w[i,:])) for i in range(N_out)]
                error_MLE_TN[:,b] = [sim.error_metric(w[i,:]/sum(w[i,:]),ss.rankdata(-pi_MLE[i,b,:],method='average')) for i in range(N_out)]
            
        # Mean over bootstrap experiments
        if rc_flag:
            MSE_RC_N.append(MSE_RC_TN)
            error_RC_N.append(error_RC_TN)
        if borda_flag:
            error_borda_N.append(error_borda_TN)
        if mle:
            MSE_MLE_N.append(MSE_MLE_TN)
            error_MLE_N.append(error_MLE_TN)
        
    if rc_flag:
        MSE_RC.append(MSE_RC_N)
        error_RC.append(error_RC_N)
    if borda_flag:
        error_borda.append(error_borda_N)
    if mle_flag:
        MSE_MLE.append(MSE_MLE_N)
        error_MLE.append(error_MLE_N)

with open("parameters_RC_N100.txt", "wb") as param:
    pickle.dump(parameters, param)

with open("MSE_RC_N100.txt", "wb") as mse_rc:
    pickle.dump(MSE_RC, mse_rc)
    
with open("MSE_MLE_N100.txt", "wb") as mse_mle:
    pickle.dump(MSE_MLE, mse_mle)
    
with open("borda_count_N100.txt", "wb") as borda_count:
    pickle.dump(error_borda, borda_count)
    
with open("error_MLE_N100.txt", "wb") as error_mle:
    pickle.dump(error_MLE, error_mle)
    
with open("error_RC_N100.txt", "wb") as error_rc:
    pickle.dump(error_RC,error_rc)
