import sys, os, csv, importlib
import numpy as np
import scipy as sc
import scipy.linalg as spl
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('/Users/eglantine.karle/Documents/GitHub/Dynamic_Rank_Centrality/modules')

import nfl_module as nfl

# Load datasets
data_dir = '/Users/eglantine.karle/Documents/GitHub/Dynamic_Rank_Centrality/nfl/nfl_data'
all_rnds = np.arange(1,17)
all_seasons = np.arange(2009,2016)

team_id = pd.read_csv(os.path.join(data_dir, "nfl_id.csv"))
elo_all = pd.read_csv(os.path.join(data_dir, "nfl_elo.csv"))

for season in all_seasons:
    random.seed(0)
    np.random.seed(0)
    result = nfl.get_final_rank_season(data_dir, season, team_id, all_rnds,elo_all,t = 1,loocv= True,num_loocv = 40,borda = True,delta_borda = False,elo = True, mle = True,loocv_mle = 200)

    df_elo,l_rc,l_borda,df_mle = result
    pi_rc,df_rc,delta_rc = l_rc
    pi_borda,df_borda,delta_borda = l_borda

    # Save results with value of delta_star in title
    df_rc.to_csv('ranks_rc_'+str(season)+'_delta_'+str(int(delta_rc))+'.csv')
    df_borda.to_csv('ranks_borda_'+str(season)+'_delta_'+str(int(delta_borda))+'.csv')
    df_elo.to_csv('ranks_elo_'+str(season)+'.csv')
    df_mle.to_csv('ranks_mle_'+str(season)+'.csv')

