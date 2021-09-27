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

## RC setup ##

def get_single_round_matrix(rnd_num, nfl_data_dir, season):
    """
    Gets the pairwise numpy array of win/loss across teams for a single
       round in a season. pwise_diff[i,j] = 1 if i won against j at this round.
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


def get_final_rank_season(data_dir, season, team_id, all_rnds,elo_all,t = 1,loocv= True,num_loocv = 200,borda = True,delta_borda = False,elo = True, mle = False,loocv_mle = 20):
    
    Y = np.array([get_single_round_matrix(rnd_num=rnd, nfl_data_dir=data_dir, season=season)[0] 
                                  for rnd in all_rnds])
    A = np.array([get_single_round_matrix(rnd_num=rnd, nfl_data_dir=data_dir, season=season)[1] 
                                  for rnd in all_rnds])
    # T-N-N arrays, T = total nb of rounds
    T, N = Y.shape[:2]
    result = []
    delta_list = np.linspace(1/2, T,10)
    
    if elo:
        df_elo = get_elo_rank_season(elo_all, season)
        
        result.append(df_elo)
    
    # Choice of delta
    if loocv:
        delta_rc,pi_rc = loocv_module.loocv_rc(Y,A,delta_list,num_loocv,t)
        
        df_rc = team_id[['name']].copy()
        ranks_rc = ss.rankdata(-pi_rc,method='average')
        df_rc.insert(1,'Rank',ranks_rc)
        df_rc.rename(columns = {'name':'RC'}, inplace = True)
        df_rc = df_rc.sort_values('Rank')
        df_rc = df_rc.set_index('Rank')
        l_rc = [pi_rc,df_rc,delta_rc]
        
    else: # Try multiple values of delta
        
        pi_rc = np.zeros((len(delta_list),N))
        l_rc = []

        for i in range(len(delta_list)):
            pi_rc[i,:] = sim.RC_dyn(t,Y,A,delta_list[i],tol = 1e-12)
            df_rc = team_id[['name']].copy()
            ranks_rc = ss.rankdata(-pi_rc[i,:],method='average')
            df_rc.insert(1,'Rank',ranks_rc)
            df_rc.rename(columns = {'name':'RC'}, inplace = True)
            df_rc = df_rc.sort_values('Rank')
            df_rc = df_rc.set_index('Rank')
            l_rc.append([pi_rc,df_rc,delta_list[i]])
        
    result.append(l_rc)
        
    if borda:
        if delta_borda:
            delta_borda,pi_borda = loocv_module.loocv_borda(Y,A,delta_list,t,num_loocv)
            
            df_borda = team_id[['name']].copy()
            ranks_borda = ss.rankdata(-pi_borda[:,],method='average')
            df_borda.insert(1,'Rank',ranks_borda)
            df_borda.rename(columns = {'name':'Borda'}, inplace = True)
            df_borda = df_borda.sort_values('Rank')
            df_borda = df_borda.set_index('Rank')
            l_borda = [pi_borda,df_borda,delta_borda]
        else:
            if loocv:
                delta_borda = delta_rc
                
                pi_borda = sim.borda_count(t,Y,A,delta_borda)
                df_borda = team_id[['name']].copy()
                ranks_borda = ss.rankdata(-pi_borda[:,],method='average')
                df_borda.insert(1,'Rank',ranks_borda)
                df_borda.rename(columns = {'name':'Borda'}, inplace = True)
                df_borda = df_borda.sort_values('Rank')
                df_borda = df_borda.set_index('Rank')
                l_borda = [pi_borda,df_borda,delta_borda]
                
            else:
                
                pi_borda = np.zeros((len(delta_list),N))
                l_borda = []

                for i in range(len(delta_list)):
                    pi_borda[i,:] = sim.borda_count(t,Y,A,delta_list[i])
                    df_borda = team_id[['name']].copy()
                    ranks_borda = ss.rankdata(-pi_borda[i,:],method='average')
                    df_borda.insert(1,'Rank',ranks_borda)
                    df_borda.rename(columns = {'name':'Borda'}, inplace = True)
                    df_borda = df_borda.sort_values('Rank')
                    df_borda = df_borda.set_index('Rank')
                    l_borda.append([pi_borda,df_borda,delta_list[i]])
        
        result.append(l_borda)
    
    if mle:
        df_mle = get_final_rank_season_mle(data_dir, season, team_id, all_rnds,True,loocv_mle, threshold = 3)
        df_mle.reset_index(inplace=True)
        df_mle.rename(columns = {'index':'MLE'}, inplace = True)
        df_mle = df_mle.sort_values('Rank')
        df_mle = df_mle.set_index('Rank')
        
        result.append(df_mle)
        
    return result
    

### MLE Method ###

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

### ELO Method ###
        
def get_elo_rank_season(elo_all, season):
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





