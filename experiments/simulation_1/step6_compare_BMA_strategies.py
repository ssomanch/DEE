import pandas as pd
import numpy as np
import torch
import argparse
import sys
import scipy
sys.path.append('../../GPCorrection/')
from src.TwoStageGPJustRBF import TwoStageGPJustRBFWrapper
from sklearn.metrics import pairwise_distances,mean_squared_error

#########################################
# Set up the experiment                 #
#########################################

parser = argparse.ArgumentParser(description='Compute PLP BMAs')
parser.add_argument('--seed1', help='Seed for sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)
parser.add_argument('--unfiltered', help='Use estimates in L, not U',action='store_true')

args = parser.parse_args()

unfiltered = args.unfiltered

#########################################
# Set up the experiment                 #
#########################################

OUTDIR = f'../output/simulation_1/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

if unfiltered:
    # Load the estimates at all points in L
    L_ests = pd.read_csv(f'{OUTDIR}/unfiltered_estimates_in_L.csv')
    train_x = torch.tensor(L_ests[['X1','X2']].values, dtype=torch.float)
    CATE_est_var = torch.tensor(L_ests.ses.values, dtype=torch.float)**2
else:
    # Load the estimates at the Voroni KNN centers
    VKNN_ests = pd.read_csv(f'{OUTDIR}/TE_bias_estimates_at_voroni_KNN_centers.csv')
    train_x = torch.tensor(VKNN_ests[['X1','X2']].values, dtype=torch.float)
    CATE_est_var = torch.tensor(VKNN_ests.ses.values, dtype=torch.float)**2

test_x1,test_x2 = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
test_x = np.concatenate([test_x1.flatten().reshape(1,-1).T,
                        test_x2.flatten().reshape(1,-1).T],
                    axis=1)
test_x = torch.from_numpy(test_x).type(torch.float)

# Load the CF test grid estimates
cf_test_CATE_est = pd.read_csv(f'{OUTDIR}/test_grid_tau_hats_from_CF.csv').cf_CATE_est

# Load the true CATEs
true_test_tau = pd.read_csv(f'{OUTDIR}/true_test_CATE_and_bias.csv').CATE
cf_test_bias = cf_test_CATE_est - true_test_tau

# Compute minimum distance to a training instance
d = pairwise_distances(test_x,train_x)
min_d = d.min(axis=1)

#########################################
# Consider four weighting strategies    #
#     - MLL weighting                   #
#     - LOO weighting                   #
#     - Weighted BLOOCV                 #
#     - 1-MC BLOOCV                     #
# But don't use weighted BLOOCV for     #
# unfiltered case                       #
#########################################

target_cols = ['bias','CATE']
results = {}
gp_test_posterior_means = {}
for t in target_cols:
    print(f'Fitting {t} GP...')
    if unfiltered:
        train_y = torch.from_numpy(L_ests[t].values).type(torch.float)
    else:
        train_y = torch.from_numpy(VKNN_ests[t].values).type(torch.float)

    gp = TwoStageGPJustRBFWrapper()
    gp.fit(train_x,train_y,CATE_est_var)
    lml = gp.get_lml(train_x,train_y,CATE_est_var)
    min_d_weighted_PLP = 0
    random_d_weighted_PLP = 0
    LOO_PLP = 0
    for ix in range(train_x.shape[0]):
        if not unfiltered:
            # Only computed weighted BLOOCV variant for filtered case
            min_d_weighted_PLP += get_weighted_plp(gp,train_x,train_y,CATE_est_var,ix,min_d,'min dist')
        LOO_PLP += get_loo_plp(gp,train_x,train_y,CATE_est_var,ix)
        random_d_weighted_PLP += random_dist_PLP_and_candidate_weights(gp,train_x,train_y,CATE_est_var,
                                                                       ix,min_d)
        
    results[t] = {
        'min dist': min_d_weighted_PLP,
        'random dist': random_d_weighted_PLP,
        'LOO': LOO_PLP,
        'MLL': lml
    }
    if unfiltered:
        del results[t]['min dist']
    gp_test_posterior_means[t] = gp.posterior(test_x).mu
    
all_results = pd.DataFrame(results)

#########################################
# Get the weighted estimators           #
#########################################

all_weighted_MSEs = {}

if unfiltered:
    strategies = ['LOO','random dist','MLL']
else:
    strategies = ['min dist','LOO','random dist','MLL']
for strategy in strategies:
    
    print(f'Computing {strategy} weighting MSE...')

    weights = scipy.special.softmax(
        np.array([results['bias'][strategy],results['CATE'][strategy]])
    )

    posterior_weighted_mean = weights[0]*(cf_test_CATE_est - gp_test_posterior_means['bias']) +\
                              weights[1]*gp_test_posterior_means['CATE']
    
    if (results['bias'][strategy] > results['CATE'][strategy]):
        zero_one_weight = cf_test_CATE_est - gp_test_posterior_means['bias']
    else:
        zero_one_weight = gp_test_posterior_means['CATE']
    
    all_weighted_MSEs[strategy] = {'Mixed MSE':mean_squared_error(posterior_weighted_mean,true_test_tau),
                                   'Percent bias in mixture':weights[0],
                                   'Zero One MSE':mean_squared_error(zero_one_weight,true_test_tau)}
    
all_weighted_MSEs = pd.DataFrame(all_weighted_MSEs)

all_results = pd.concat([all_results,all_weighted_MSEs.T],axis=1)

#########################################
# Dump results                          #
#########################################
print(all_results)
if unfiltered:
    all_results.to_csv(f"{OUTDIR}/PLP_strategy_search_unfiltered.csv")
else:
    all_results.to_csv(f"{OUTDIR}/PLP_strategy_search.csv")
