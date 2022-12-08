import pandas as pd
import numpy as np
import torch
import os
import sys
import scipy
sys.path.append(os.getcwd())
from DeeGPs.TwoStageGPJustRBF import TwoStageGPJustRBFWrapper

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid


import argparse

parser = argparse.ArgumentParser(description='Fit GPs to CATE and bias estimates in U.')
parser.add_argument('--seed1', help='Seed for sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)
parser.add_argument('--unfiltered', help='Use estimates in L, not U',action='store_true')
parser.add_argument('--alpha', default = 0.1, type=float, 
                    help='significance level for confidence region')

args = parser.parse_args()

unfiltered = args.unfiltered

OUTDIR  = f'output/simulation_1/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

###############################################
# Utilities                                   #
###############################################

def get_GP_model_outputs(train_x,train_y,CATE_est_var,GPmodel,target,test_x, model_name, alpha=0.1, 
                         cf_target_est = None,
                         cf_target_var = None):
    # JUST FOR EXPERIMENTATION: NEED TO BE REMOVED
    # index_var_less_5 = (CATE_est_var <5).nonzero().transpose(0,1)[0]
    # CATE_est_var = CATE_est_var[index_var_less_5]
    # train_x = train_x[index_var_less_5]
    # train_y = train_y[index_var_less_5]
    #CATE_est_var = torch.clamp(CATE_est_var, max=5)
    #print( np.mean(CATE_est_var.numpy()) )
    #CATE_est_var = torch.tensor(np.repeat(np.mean(CATE_est_var.numpy()), len(CATE_est_var)),dtype=torch.float)
    

    # Step 1: fit model
    model = GPmodel()
    model.fit(train_x,train_y,CATE_est_var)

    # Step 2: get posterior
    posterior = model.posterior(test_x)
    lml_train = model.get_lml(train_x,train_y,CATE_est_var)
    
    # Step 3: Get the MSE over the test set:
    mse = np.mean((posterior.mu.values - target.values)**2)
    
    # # Step 4: Get coverage probability
    # posterior_predictive_df = model.posterior_predictive(test_x)
    # # By defualt upper and lower returns two(2) standard deviations above and below the mean.
    # # https://docs.gpytorch.ai/en/v1.6.0/_modules/gpytorch/distributions/multivariate_normal.html
    # temp_lower, temp_upper = posterior.lower,  posterior.upper
    # std_dev = (temp_upper - temp_lower)/(2*2)
    # print("Posterior Standard Error")
    # print((np.mean(std_dev**2))**(0.5))
    # #print(std_dev.describe())
    
    # ## Assuming normal distribution at each prediction 
    # z_alpha_by_2 = scipy.stats.norm.ppf(1-alpha/2)
    # if (model_name == "bias"):
    #     temp_est = cf_target_est - posterior.mu.values
    #     temp_target = cf_target_est - target.values
    #     temp_std = np.sqrt(std_dev**2 + cf_target_var)
    #     cover_prob_post = np.mean((temp_est - z_alpha_by_2*temp_std < temp_target) &
    #                               (temp_target < temp_est + z_alpha_by_2*temp_std))
    # else:
    #     cover_prob_post = np.mean((posterior.mu.values - z_alpha_by_2*std_dev < target.values) & 
    #                               (target.values < posterior.mu.values + z_alpha_by_2*std_dev))
    
    # print(f"Posterior Coverage Probability is {cover_prob_post}")
    
    # temp_lower, temp_upper = posterior_predictive_df.lower,  posterior_predictive_df.upper
    # std_dev = (temp_upper - temp_lower)/(2*2)
    # print("Posterior Predictive Standard Error")
    # print((np.mean(std_dev**2))**(0.5))
    # #print(std_dev.describe())
    
    # # Assuming normal distribution at each prediction 
    # z_alpha_by_2 = scipy.stats.norm.ppf(1-alpha/2)
    # cover_prob_pred = np.mean((posterior.mu.values - z_alpha_by_2*std_dev < target.values) & 
    #                      (target.values < posterior.mu.values + z_alpha_by_2*std_dev))
    
    # print(f"Posterior Predictive Coverage Probability is {cover_prob_pred}")
    
    
    
    # Step 5: Save the test predictions to a file
    # TODO: Need to use the DFs directly coming from posterior and posterior predictive and merge, 
    # instead of recreating them. 
    
    test_df_to_save = pd.DataFrame(test_x.numpy())
    test_df_to_save.columns = ['X1', 'X2']
    test_df_to_save['posterior_mu'] = posterior.mu.values
    # test_df_to_save['post_pred_lower'] = posterior_predictive_df.lower
    # test_df_to_save['post_pred_upper'] = posterior_predictive_df.upper
    # test_df_to_save['post_lower'] = posterior.lower
    # test_df_to_save['post_upper'] = posterior.upper
    test_df_to_save['target'] = target.values
    
    if unfiltered:
        test_df_to_save.to_csv(f"{OUTDIR}/{model_name}_test_predictions_unfiltered.csv", index=False)
    else:
        test_df_to_save.to_csv(f"{OUTDIR}/{model_name}_test_predictions.csv", index=False)
        
    # temp_lower, temp_upper = posterior.lower,  posterior.upper
    # std_dev = (temp_upper - temp_lower)/(2*2)
    # print("Posterior Standard Error")
    # print((np.mean(std_dev**2))**(0.5))
    
    # noise_df = model.noise_posterior(test_x)
    # temp_lower, temp_upper = noise_df.lower,  noise_df.upper
    # std_dev = (temp_upper - temp_lower)/(2*2)
    # print("Noise Posterior Standard Error")
    # print((np.mean(noise_df.mu))**(0.5))
    
    # print("Noise Prior Standard Error")
    # print(np.mean(CATE_est_var.numpy())**0.5)
    
    #print(pd.Series(target.values).describe())
    #print(pd.Series(posterior.mu.values).describe())
    
    

    # Compute Negative Log Predictive Density (NLPD) and Mean Standardized Log Loss (MSLL)
    # print(model.get_posterior_predictive_metrics(test_x, target.values))
       
    return {'MSE':mse,
            'Description':model.describe(),
            'LML':lml_train}

def get_observational_estimator_scores(model_obs_est,true_test_tau):

    # Obs Est MSE
    mse = np.mean((model_obs_est - true_test_tau)**2)

    return {'MSE':mse,
            'Description':{}
            }

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
    # TODO: Remove. This is just for testing need to be removed
    #train_CATE_est = torch.tensor(L_ests.true_CATE.values, dtype=torch.float)
    #CATE_est_var = torch.tensor(np.repeat(0.01, len(L_ests.ses.values)), dtype=torch.float)**2
else:
    # Load the estimates at the Voroni KNN centers
    VKNN_ests = pd.read_csv(f'{OUTDIR}/TE_bias_estimates_at_voroni_KNN_centers.csv')
    train_x = torch.tensor(VKNN_ests[['X1','X2']].values, dtype=torch.float)
    train_bias_est = torch.tensor(VKNN_ests.bias.values, dtype=torch.float)
    train_CATE_est = torch.tensor(VKNN_ests.CATE.values, dtype=torch.float)
    CATE_est_var = torch.tensor(VKNN_ests.ses.values, dtype=torch.float)**2
    # TODO: Remove. This is just for testing need to be removed
    #train_CATE_est = torch.tensor(VKNN_ests.true_CATE.values, dtype=torch.float)
    #CATE_est_var = torch.tensor(np.repeat(0.01, len(VKNN_ests.ses.values)), dtype=torch.float)**2

# Set up the test Xs
# TODO: Probably read from test_grid.csv instead of recreating the test meshgrid again??
test_x1,test_x2 = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
test_x = np.concatenate([test_x1.flatten().reshape(1,-1).T,
                        test_x2.flatten().reshape(1,-1).T],
                    axis=1)
test_x = torch.tensor(test_x, dtype=torch.float)

# Load the CF test grid estimates
cf_test_CATE_est = pd.read_csv(f'{OUTDIR}/test_grid_tau_hats_from_CF.csv').cf_CATE_est
cf_test_CATE_var = pd.read_csv(f'{OUTDIR}/test_grid_tau_hats_from_CF.csv').cf_CATE_var

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
    train_x,train_y,CATE_est_var,TwoStageGPJustRBFWrapper,model_target,test_x, "bias", args.alpha, 
    cf_target_est = cf_test_CATE_est,
    cf_target_var =cf_test_CATE_var
)

save_outputs(r,OUTDIR,'bias_correction',unfiltered)

###############################################
# Fit the direct extrapolation model and save #
###############################################

print('Fitting direct extrapolation model...')
train_y = train_CATE_est
model_target = true_test_tau

r = get_GP_model_outputs(
    train_x,train_y,CATE_est_var,TwoStageGPJustRBFWrapper,model_target,test_x, "CATE", args.alpha
)
save_outputs(r,OUTDIR,'CATE_extrapolation',unfiltered)

################################################
# Save scores ignoring the RD estimates        #
################################################

print('Scoring CF estimation...')
obs_est_scores = get_observational_estimator_scores(cf_test_CATE_est,true_test_tau)
save_outputs(obs_est_scores,OUTDIR,'cf_ignoring_RDs',unfiltered)
