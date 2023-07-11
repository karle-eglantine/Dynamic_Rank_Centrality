# Dynamic Rank Centrality
 This repository contains the code for the paper Dynamic Ranking with the BTL Model: A Nearest Neighbor based Rank Centrality Method. https://arxiv.org/abs/2109.13743
 
## Files

+ Plots_synthetic_data.ipynb is a notebook that contains the script to generate Figures 1,2,3,4 and Tables 1,2 (experiments on synthetic data). By default, the notebook runs toy examples to provide a good understanding of the functions to the user. The parameters used in the paper need to be uncommented to reproduce the figures of the article
+ NFL_analysis.ipyng is a notebook that contains the analysis of the NFL dataset (available in the nflWAR package), leading to Figure 5 and Tables 3,4.
+ drc.py contains a function to generate synthetic data and perform their analysis under the Dynamic BTL model. It provides the $\ell_2,\ell_\infty$ and $D_{\pi^*}(\sigma)$ errors as well as running times for each methods.
+ performance.py contains a function to perform DRC method for different parameters $\delta$.
+ dynamic_btl.yml : conda environment
+ modules is a directory containing scripts of some technical tools.
+ Figures is a foldeer containing all the figures presented in the article.



 


