import pandas as pd
import numpy as np
import torch
import argparse
import os
import sys
sys.path.append(os.getcwd())
import scipy
from DeeGPs.TwoStageGPJustRBF import TwoStageGPJustRBFWrapper
from DeeGPs.PLP_BMA_utils import *
from sklearn.metrics import pairwise_distances,mean_squared_error

#########################################
# Set up the experiment                 #
#########################################

parser = argparse.ArgumentParser(description='Compute PLP BMAs')
parser.add_argument('--seed1', help='Seed for sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)
parser.add_argument('--unfiltered', help='Use estimates in L, not U',action='store_true')
parser.add_argument('--alpha', default = 0.1, type=float, 
                    help='significance level for confidence region')

args = parser.parse_args()

unfiltered = args.unfiltered


def contains_true_val(mean_pred, se_pred, true_val, alpha = 0.1):
    '''
    Checks if the true_val is with in the confidence interval based on mean_pred and se_pred
    '''
    ## Assuming normal distribution at each prediction 
    z_alpha_by_2 = scipy.stats.norm.ppf(1-alpha/2)
    return ((mean_pred - z_alpha_by_2*se_pred < true_val) & (true_val < mean_pred + z_alpha_by_2*se_pred))
    
#########################################
# Set up the experiment                 #
#########################################

OUTDIR = f'output/simulation_1/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

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
cf_test_CATE_df = pd.read_csv(f'{OUTDIR}/test_grid_tau_hats_from_CF.csv')
cf_test_CATE_est = cf_test_CATE_df.cf_CATE_est
cf_test_CATE_var = cf_test_CATE_df.cf_CATE_var
cf_test_CATE_se = np.sqrt(cf_test_CATE_var)

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
gp_test_posterior_se = {}
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
    posterior_df = gp.posterior(test_x)
    
    gp_test_posterior_means[t] = posterior_df.mu
    
    posterior_predictive_df = gp.posterior_predictive(test_x)
    # By defualt upper and lower returns two(2) standard deviations above and below the mean.
    # https://docs.gpytorch.ai/en/v1.6.0/_modules/gpytorch/distributions/multivariate_normal.html
    temp_lower, temp_upper = posterior_predictive_df.lower,  posterior_predictive_df.upper
    gp_test_posterior_se[t] = (temp_upper - temp_lower)/(2*2)
    
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
    
    ## Assuming independence of causal forest estimate, posterior se for bias, and posterior se for cate
    # posterior_weighted_se = np.sqrt((weights[0]**2)*(cf_test_CATE_se**2 + gp_test_posterior_se['bias']**2) +\
    #                                 (weights[1]**2)*(gp_test_posterior_se['CATE']**2))
    posterior_weighted_se = np.sqrt((weights[0]**2)*(gp_test_posterior_se['bias']**2) +\
                                    (weights[1]**2)*(gp_test_posterior_se['CATE']**2))
    
    if (results['bias'][strategy] > results['CATE'][strategy]):
        zero_one_weight = cf_test_CATE_est - gp_test_posterior_means['bias']
        #zero_one_weight_se = np.sqrt(cf_test_CATE_se**2 + gp_test_posterior_se['bias']**2)
        zero_one_weight_se = gp_test_posterior_se['bias']
    else:
        zero_one_weight = gp_test_posterior_means['CATE']
        zero_one_weight_se = gp_test_posterior_se['CATE']
    
    all_weighted_MSEs[strategy] = {'Mixed MSE':mean_squared_error(posterior_weighted_mean,true_test_tau),
                                   'Percent bias in mixture':weights[0],
                                   'Zero One MSE':mean_squared_error(zero_one_weight,true_test_tau),
                                   'Mixed CP': np.mean(contains_true_val(posterior_weighted_mean, 
                                                                         posterior_weighted_se, 
                                                                         true_test_tau, 
                                                                         alpha=args.alpha)), 
                                   'Zero One CP': np.mean(contains_true_val(zero_one_weight, 
                                                                            zero_one_weight_se, 
                                                                            true_test_tau, 
                                                                            alpha=args.alpha))}
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
