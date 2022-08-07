import pandas as pd
import numpy as np
import torch
import os
import sys
sys.path.append('../../GPCorrection/')
from src.TwoStageGPJustRBF import TwoStageGPJustRBFWrapper

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid


import argparse

parser = argparse.ArgumentParser(description='Fit GPs to CATE and bias estimates in U.')
parser.add_argument('--seed1', help='Seed for sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)
parser.add_argument('--unfiltered', help='Use estimates in L, not U',action='store_true')

args = parser.parse_args()

unfiltered = args.unfiltered

OUTDIR  = f'../output/simulation_1/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

###############################################
# Utilities                                   #
###############################################

def get_GP_model_outputs(train_x,train_y,CATE_est_var,GPmodel,target,test_x):
    # Step 1: fit model
    model = GPmodel()
    model.fit(train_x,train_y,CATE_est_var)

    # Step 2: get posterior
    posterior = model.posterior(test_x)
    lml_train = model.get_lml(train_x,train_y,CATE_est_var)
    
    # Step 3: Get the MSE over the test set:
    mse = np.mean((posterior.mu.values - target.values)**2)
    
    return {'MSE':mse,
            'Description':model.describe(),
            'LML':lml_train}

def get_observational_estimator_scores(model_obs_est,true_test_tau):

    # Obs Est MSE
    mse = np.mean((model_obs_est - true_test_tau)**2)

    return {'MSE':mse,
            'Description':{}}

def save_outputs(model_results,root_dir,model_name,unfiltered):
    
    model_results['Description']['MSE'] = model_results['MSE']
    
    if model_name != 'cf_ignoring_RDs':
        model_results['Description']['LML'] = model_results['LML']

    output = pd.Series(model_results['Description'],name='Value')
    output.index.name = 'Variable'
    
    if unfiltered:
        output.to_csv(f"{root_dir}/{model_name}_description_unfiltered.csv")
    else:
        output.to_csv(f"{root_dir}/{model_name}_description.csv")



################################################
# Load and set up the data 
################################################

if unfiltered:
    # Load the estimates at all points in L
    L_ests = pd.read_csv(f'{OUTDIR}/unfiltered_estimates_in_L.csv')
    train_x = torch.tensor(L_ests[['X1','X2']].values, dtype=torch.float)
    train_bias_est = torch.tensor(L_ests.bias.values, dtype=torch.float)
    train_CATE_est = torch.tensor(L_ests.CATE.values, dtype=torch.float)
    CATE_est_var = torch.tensor(L_ests.ses.values, dtype=torch.float)**2
else:
    # Load the estimates at the Voroni KNN centers
    VKNN_ests = pd.read_csv(f'{OUTDIR}/TE_bias_estimates_at_voroni_KNN_centers.csv')
    train_x = torch.tensor(VKNN_ests[['X1','X2']].values, dtype=torch.float)
    train_bias_est = torch.tensor(VKNN_ests.bias.values, dtype=torch.float)
    train_CATE_est = torch.tensor(VKNN_ests.CATE.values, dtype=torch.float)
    CATE_est_var = torch.tensor(VKNN_ests.ses.values, dtype=torch.float)**2

# Set up the test Xs
test_x1,test_x2 = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
test_x = np.concatenate([test_x1.flatten().reshape(1,-1).T,
                        test_x2.flatten().reshape(1,-1).T],
                    axis=1)
test_x = torch.tensor(test_x, dtype=torch.float)

# Load the CF test grid estimates
cf_test_CATE_est = pd.read_csv(f'{OUTDIR}/test_grid_tau_hats_from_CF.csv').cf_CATE_est

# Load the true test CATEs
true_test_tau = pd.read_csv(f'{OUTDIR}/true_test_CATE_and_bias.csv').CATE
cf_test_bias = cf_test_CATE_est - true_test_tau

###############################################
# Fit the bias correcting model + save output #
###############################################

print('Fitting bias correcting model...')
train_y = train_bias_est
model_target = cf_test_bias
    
r = get_GP_model_outputs(
    train_x,train_y,CATE_est_var,TwoStageGPJustRBFWrapper,model_target,test_x
)

save_outputs(r,OUTDIR,'bias_correction',unfiltered)

###############################################
# Fit the direct extrapolation model and save #
###############################################

print('Fitting direct extrapolation model...')
train_y = train_CATE_est
model_target = true_test_tau

r = get_GP_model_outputs(
    train_x,train_y,CATE_est_var,TwoStageGPJustRBFWrapper,model_target,test_x
)
save_outputs(r,OUTDIR,'CATE_extrapolation',unfiltered)

################################################
# Save scores ignoring the RD estimates        #
################################################

print('Scoring CF estimation...')
obs_est_scores = get_observational_estimator_scores(cf_test_CATE_est,true_test_tau)
save_outputs(obs_est_scores,OUTDIR,'cf_ignoring_RDs',unfiltered)