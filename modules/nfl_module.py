# This module contains useful functions for the analysis of the NFL dataset.

import sys, os, csv, importlib
import numpy as np
import scipy as sc
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt

import simulation_module as sim
import grad_module as model
import mle_module as mle
import loocv_module

# DRC setup

def get_single_round_matrix(rnd_num, nfl_data_dir, season):
    """
    Gets the pairwise numpy array Y of win/loss across teams for a single
       round in a season and the corresponding adjacency matrix A. 
       Y[i,j] = 1 if j won against j at this round and A[i,j] = 1 if i and j played against one another at this round.
    """
    fname = "round" + "_" + str(rnd_num).zfill(2) + ".csv"
    fpath = os.path.join(nfl_data_dir, str(season), fname)
    rnd_df = pd.read_csv(fpath)
    
    # Matrix of data Y
    Y = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    Y[Y >= 0] = 0
    Y[Y < 0] = 1
    Y[np.isnan(Y)] = 0
    
    # Adjacency matrix A
    A = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    A[np.isnan(A)] = 0
    A[~np.isnan(A)] = 1
    return Y,A


def get_final_rank_season(data_dir, season, team_id, all_rnds,elo_all,t = 1,loocv= True,num_loocv = 40,borda = True,do_loocv_borda = False,
                          elo = True, mle = False,loocv_mle = 20):
    '''
    data_dir: directory of the nfl data
    season: season for which we want to recover the ranks
    team_id: dataframe of the team names
    all_rnds : array of the rounds to consider for the estimation (usually all_rnds = np.arange(1,17) )
    elo_all : elo ratings
    t : time at which we want to recover the ranks (end of the season : t=1)
    loocv : boolean indicating if loocv is performed to estimate delta for DRC and Borda Count methods.
    num_loocv : nimber of loocv to perform 
    borda : boolean indicating if Borda Count method is performed
    do_loocv_borda : indicate if we perform the loocv adapted to Borda method (default value is False, we choose the same delta for DRC and Borda Count)
    elo : indicates if we estimate the ranks from the elo ratings
    mle : indicates if we estimate the ranks for the MLE method
    loocv_mle : indicates the number of loocv for the MLE method
    --------
    Get dataframes of the ranking of the teams at time t during a season, for ELO,DRC,MLE and Borda count methods.
    '''
    
    # Load the data for each round as a data matrix Y and an adjacency matrix A
    Y = np.array([get_single_round_matrix(rnd_num=rnd, nfl_data_dir=data_dir, season=season)[0] 
                                  for rnd in all_rnds])
    A = np.array([get_single_round_matrix(rnd_num=rnd, nfl_data_dir=data_dir, season=season)[1] 
                                  for rnd in all_rnds])
    T, N = Y.shape[:2]
    result = []
    # List of candidates for delta in the loocv
    delta_list = np.linspace(1/2, T,10)
    
    if elo:
        df_elo = get_elo_rank_season(elo_all, season)
        
        result.append(df_elo)
    
    # Choice of delta for the DRC and estimation of the ranks
    if loocv:
        delta_rc,pi_rc = loocv_module.loocv_rc(Y,A,delta_list,num_loocv,t)
        
        df_rc = team_id[['name']].copy()
        # Order the teams from their estimated strengths
        ranks_rc = ss.rankdata(-pi_rc,method='average')
        df_rc.insert(1,'Rank',ranks_rc)
        df_rc.rename(columns = {'name':'RC'}, inplace = True)
        df_rc = df_rc.sort_values('Rank')
        df_rc = df_rc.set_index('Rank')
        # Save the estimation of weights, the optimal value of delta and the dataframe of the ranks
        l_rc = [pi_rc,df_rc,delta_rc]
        
    else: # Try multiple values of delta
        
        pi_rc = np.zeros((len(delta_list),N))
        l_rc = []

        for i in range(len(delta_list)):
            # Estimate the strengths/ranks for each value of delta
            pi_rc[i,:] = sim.RC_dyn(t,Y,A,delta_list[i],tol = 1e-12)
            df_rc = team_id[['name']].copy()
            ranks_rc = ss.rankdata(-pi_rc[i,:],method='average')
            df_rc.insert(1,'Rank',ranks_rc)
            df_rc.rename(columns = {'name':'RC'}, inplace = True)
            df_rc = df_rc.sort_values('Rank')
            df_rc = df_rc.set_index('Rank')
            # Save for each value of delta the estimation of the weights and the associated datatfram of the ranks
            l_rc.append([pi_rc,df_rc,delta_list[i]])
        
    result.append(l_rc)
        
    if borda:
        if do_loocv_borda: # Perform another loocv adapted to the Borda Count Method
            # Estimate the optimal value of delta and the associated win rates
            delta_borda,pi_borda = loocv_module.loocv_borda(Y,A,delta_list,t,num_loocv)
            # Creation of the dataframe of the ranks
            df_borda = team_id[['name']].copy()
            ranks_borda = ss.rankdata(-pi_borda[:,],method='average')
            df_borda.insert(1,'Rank',ranks_borda)
            df_borda.rename(columns = {'name':'Borda'}, inplace = True)
            df_borda = df_borda.sort_values('Rank')
            df_borda = df_borda.set_index('Rank')
            # Save the estimation of the win rates, of the ranks and the value of delta
            l_borda = [pi_borda,df_borda,delta_borda]
        else:
            if loocv: # Use the value of delta obtained by loocv for the DRC method
                delta_borda = delta_rc
                # Estimation for this given delta
                pi_borda = sim.borda_count(t,Y,A,delta_borda)
                # Estimation of the ranks from the strenghts
                df_borda = team_id[['name']].copy()
                ranks_borda = ss.rankdata(-pi_borda[:,],method='average')
                df_borda.insert(1,'Rank',ranks_borda)
                df_borda.rename(columns = {'name':'Borda'}, inplace = True)
                df_borda = df_borda.sort_values('Rank')
                df_borda = df_borda.set_index('Rank')
                # Save the esimated strengths, ranks and the optimal value of delta
                l_borda = [pi_borda,df_borda,delta_borda]
                
            else:
                # Estimate the ranks for different values of delta
                pi_borda = np.zeros((len(delta_list),N))
                l_borda = []

                for i in range(len(delta_list)):
                    # Estimate the win rates for each value of delta
                    pi_borda[i,:] = sim.borda_count(t,Y,A,delta_list[i])
                    # Creation od the dataframe of the ranks
                    df_borda = team_id[['name']].copy()
                    ranks_borda = ss.rankdata(-pi_borda[i,:],method='average')
                    df_borda.insert(1,'Rank',ranks_borda)
                    df_borda.rename(columns = {'name':'Borda'}, inplace = True)
                    df_borda = df_borda.sort_values('Rank')
                    df_borda = df_borda.set_index('Rank')
                    # Save the esimated win rates, ranks and the associated value of delta
                    l_borda.append([pi_borda,df_borda,delta_list[i]])
        
        result.append(l_borda)
    
    if mle:
        # Get the mle estimation of the ranks using the code of Bong et al.
        df_mle = get_final_rank_season_mle(data_dir, season, team_id, all_rnds,True,loocv_mle, threshold = 3)
        df_mle.reset_index(inplace=True)
        df_mle.rename(columns = {'index':'MLE'}, inplace = True)
        df_mle = df_mle.sort_values('Rank')
        df_mle = df_mle.set_index('Rank')
        
        result.append(df_mle)
        
    return result
    

# MLE Method
# This part of the code belongs to Bong et al.

def get_single_round_mle(rnd_num, nfl_data_dir, season):
    """
    Gets the pairwise numpy array of score diffences across teams for a single
       round in a season
    """
    fname = "round" + "_" + str(rnd_num).zfill(2) + ".csv"
    fpath = os.path.join(nfl_data_dir, str(season), fname)
    rnd_df = pd.read_csv(fpath)
    pwise_diff = rnd_df.pivot(index='team', columns='team_other',values='diff').values
    pwise_diff[pwise_diff >= 0] = 1
    pwise_diff[pwise_diff < 0] = 0
    pwise_diff[np.isnan(pwise_diff)] = 0
    return pwise_diff



def get_final_rank_season_mle(data_dir, season, team_id, all_rnds,loocv= True, num_loocv = 20,threshold = 3):
    game_matrix_list = np.array([get_single_round_mle(rnd_num=rnd, nfl_data_dir=data_dir, season=season) 
                                  for rnd in all_rnds])
    
    T, N = game_matrix_list.shape[:2]
    if loocv:
        h_list = np.linspace(0.5, 0.05, 15)
        h_star, nll_cv, beta = loocv_module.loocv_mle(game_matrix_list, h_list,
                                                 mle.gd_bt, num_loocv = num_loocv, return_prob = False,
                                                 verbose='cv', out='notebook')    
    else:
        h_list = np.linspace(0.5, 0.05, 15)
        val_list = []

        data = game_matrix_list
        for i in range(len(h_list)):
            h = h_list[i]
            ks_data = mle.kernel_smooth(data,h)
            val_list.append(max_change(_, beta = mle.gd_bt(ks_data,l_penalty = 0)))

        # plt.plot(lam_list,val_list)

        while val_list[-1] > threshold:
            threshold += 1

        ix = next(idx for idx, value in enumerate(val_list) if value <= threshold)
        h_star = h_list[ix]
        
        ks_data = mle.kernel_smooth(data,h_star)
        beta = mle.pgd_l2_sq(ks_data,l_penalty = 0)

    arg = np.argsort(-beta,axis=1)
    rank_list = pd.DataFrame(data={(i):team_id['name'][arg[i-1,]].values for i in range(1,2)})
    rank_last = rank_list[1]
    rank_last = pd.DataFrame({'Rank':range(len(rank_last))},index = rank_last.values)
    
    print("season " + str(season) + " finished. \n")
    
    return rank_last.sort_index() + 1

# ELO Method
        
def get_elo_rank_season(elo_all, season):
    '''
    Estimate the ranks at the end of the season from the elo ratings. 
    '''
    elo_season = elo_all.iloc[np.where(elo_all['season'] == season)]
    elo_season = elo_season[pd.isnull(elo_season['playoff'])]
    a = elo_season[['team1','elo1_post']]
    a.columns = ['team','elo']
    a = a.reset_index()
    b = elo_season[['team2','elo2_post']]
    b.columns = ['team','elo']
    b = b.reset_index()

    c = pd.concat([a,b])
    c = c.sort_values(by = ['index'])    
    d = c.groupby(by = ['team']).last()
    
    x = d.index.values
    x[np.where(x == 'LAR')] = 'STL'
    x[np.where(x == 'LAC')] = 'SD'
    x[np.where(x == 'JAX')] = 'JAC'
    x[np.where(x == 'WSH')] = 'WAS'
    
    elo_rank = pd.DataFrame({'ELO': x},index = ss.rankdata(-d['elo'])).sort_index()
    
    return elo_rank



