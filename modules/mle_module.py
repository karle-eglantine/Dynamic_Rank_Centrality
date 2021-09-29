# This module belongs to Bong et al. (2020)
# It contains useful functions to conduct the kernel smoothing and the MLE analysis

import scipy.linalg as spl
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys

import grad_module as model

def kernel_function(t,tk,h):
    # tk can be a sequence
    return 1/((2 * np.pi)**0.5 * h) * np.exp( - (t - tk)**2 / (2 * h**2))

def kernel_smooth(game_matrix_list,h,T_list = None):
    '''
    return the smoothed version of the data
    '''
    T, N = game_matrix_list.shape[0:2]
    if T_list is None:
        T_list = np.arange(T)
    smoothed = np.zeros((len(T_list),N,N)) + 0
    for s,t in enumerate(T_list):
        tt = (t + 1) / T
        tk = (np.arange(T) + 1) / T
        weight = kernel_function(tt,tk,h)
        for i in range(N):
            for j in range(N):
                smoothed[s,i,j] = sum(weight * game_matrix_list[:,i,j])/sum(weight)
    return smoothed

def gd_bt(data,
              max_iter=1000, ths=1e-12,
              step_init=0.5, max_back=200, a=0.2, b=0.5,
              beta_init=None, verbose=False, out=sys.stdout):
    '''
    Perform the MLE estimation where data has been smoothed beforehand.
    Returns a tuple (a,b) where b is the estimation of the weights at each time data where provided.
    '''
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = beta_init
    nll = model.neg_log_like(beta, data)
    # initialize record
    objective_wback = [nll]
    if verbose:
        out.write("initial objective value: %f\n"%objective_wback[-1])
        out.flush()
    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = model.grad_nl(beta, data).reshape([T,N])
        # backtracking line search
        s = step_init
        for j in range(max_back):
            beta_new = beta - s*gradient
            beta_diff = beta_new - beta
            nll_new = model.neg_log_like(beta_new, data)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))            
            if nll_new <= nll_back:
                break
            s *= b       
        # proximal gradient update
        beta = beta_new
        nll = nll_new
        # record objective value
        objective_wback.append(model.neg_log_like(beta, data))        
        #if verbose:
            #out.write("%d-th GD, objective value: %f\n"%(i+1, objective_wback[-1]))
            #out.flush()
        if abs(objective_wback[-2] - objective_wback[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()
    return objective_wback, beta

    
def objective_l2_sq(beta, game_matrix_list, l_penalty):
    '''
    compute the objective of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_penalty = np.sum(np.square(beta[:-1]-beta[1:]))
    
    return model.neg_log_like(beta, game_matrix_list) + l_penalty * l2_penalty


def grad_l2_sq(beta, game_matrix_list, l):
    '''
    compute the gradient of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_grad = model.grad_nl(beta, game_matrix_list)
    l2_grad[N:] += l * 2 * ((beta[1:]-beta[:-1])).reshape(((T - 1) * N, 1))
    l2_grad[:-N] += l * 2 *((beta[:-1]-beta[1:])).reshape(((T - 1) * N, 1))
    
    return  l2_grad


def hess_l2_sq(beta, game_matrix_list, l):
    '''
    compute the Hessian of the model (neg_log_like + l2_square)
    ----------
    Input:
    beta: TxN array or a TN vector
    game_matrix_list: TxNxN array
    ----------
    Output:
    objective: negative log likelihood + squared l2 penalty
    '''
    # reshape beta into TxN array
    T, N = game_matrix_list.shape[0:2]
    beta = np.reshape(beta, [T,N])
    
    # compute l2 penalty
    l2_hess = model.hess_nl(beta, game_matrix_list)
    off_diag = np.array([2] + [0] * (N - 1) + [-1] + [0] * (N * (T - 1) - 1))
    l2_hess += l * 2 * sc.linalg.toeplitz(off_diag,off_diag)
    l2_hess[0:N,0:N] -= l * 2 * np.diag(np.ones(N))
    l2_hess[-N:,-N:] -= l * 2 * np.diag(np.ones(N))
    return  l2_hess

def prox_l2_sq(beta, s, l):
    '''
    proximal operator for l2-square-penalty
    '''
    n = np.array(beta).shape[0]
    
    # define banded matrix
    banded = np.block([
        [np.zeros([1,1]), (-1)*2*s*l*np.ones([1,n-1])],
        [(1+2*s*l)*np.ones([1,1]), (1+2*2*s*l)*np.ones([1,n-2]), (1+2*s*l)*np.ones([1,1])],
        [(-1)*2*s*l*np.ones([1,n-1]), np.zeros([1,1])]
    ])

    # solve banded @ beta* = beta
    return spl.solve_banded((1,1), banded, beta, True, True, False)

def newton_l2_sq(data, l_penalty=1,
                 max_iter=1000, ths=1e-12,
                 step_init=1, max_back=200, a=0.01, b=0.3,
                 beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2]).reshape((N * T,1))
    else:
        beta = beta_init.reshape((N*T, 1))
    
    # initialize record
    objective_nt = [objective_l2_sq(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_nt[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = grad_l2_sq(beta, data, l_penalty)[1:]
        hessian = hess_l2_sq(beta, data, l_penalty)[1:,1:]
        # backtracking
        obj_old = np.inf
        s = step_init
        beta_new = beta - 0 # make a copy
        
        for j in range(max_back):
            v = -sc.linalg.solve(hessian, gradient)
            beta_new[1:] = beta_new[1:] + s * v
            obj_new = objective_l2_sq(beta_new,data,l_penalty)
        
            if obj_new <= obj_old + b * s * gradient.T @ v:
                break
            s *= a
            
        beta = beta_new
        
        # objective value
        objective_nt.append(obj_new)
        obj_old = obj_new

        if verbose:
            out.write("%d-th Newton, objective value: %f\n"%(i+1, objective_nt[-1]))
            out.flush()
        if objective_nt[-2] - objective_nt[-1] < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    beta = beta.reshape((T,N))
    beta = beta - sum(beta[0,0:N]) / N   

    return objective_nt, beta

def pgd_l2_sq(data, l_penalty=1,
              max_iter=1000, ths=1e-12,
              step_init=0.5, max_back=200, a=0.2, b=0.5,
              beta_init=None, verbose=False, out=sys.stdout):
    # initialize optimization
    T, N = data.shape[0:2]
    if beta_init is None:
        beta = np.zeros(data.shape[:2])
    else:
        beta = beta_init
    nll = model.neg_log_like(beta, data)

    # initialize record
    objective_wback = [objective_l2_sq(beta, data, l_penalty)]
    if verbose:
        out.write("initial objective value: %f\n"%objective_wback[-1])
        out.flush()

    # iteration
    for i in range(max_iter):
        # compute gradient
        gradient = model.grad_nl(beta, data).reshape([T,N])
        
        # backtracking line search
        s = step_init
        for j in range(max_back):
            beta_new = prox_l2_sq(beta - s*gradient, s, l_penalty)
            beta_diff = beta_new - beta
            
            nll_new = model.neg_log_like(beta_new, data)
            nll_back = (nll + np.sum(gradient * beta_diff) 
                        + np.sum(np.square(beta_diff)) / (2*s))
            
            if nll_new <= nll_back:
                break
            s *= b
        
        # proximal gradient update
        beta = beta_new
        nll = nll_new
        
        # record objective value
        objective_wback.append(objective_l2_sq(beta, data, l_penalty))
        
        if verbose:
            out.write("%d-th PGD, objective value: %f\n"%(i+1, objective_wback[-1]))
            out.flush()
        if abs(objective_wback[-2] - objective_wback[-1]) < ths:
            if verbose:
                out.write("Converged!\n")
                out.flush()
            break
        elif i >= max_iter-1:
            if verbose:
                out.write("Not converged.\n")
                out.flush()

    return objective_wback, beta    
