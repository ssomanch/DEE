library(data.table)
library(FNN)
library(LORD3)
library(grf)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)

options(dplyr.summarise.inform = FALSE)
source('utils/voroni_knn.R')

degree = 4
k = 200
M_prime = 400
k_prime = 1000
t_partition = 30
estimator = 'rotated_2SLS'
nvec_col_names = c('raw_nvec1','raw_nvec2')

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]

##########################################################
# Load data
##########################################################

OUTDIR = paste0('output/simulation_1/',seed1,'/',CATE_ls,'/',bias_ls,'/')

# The dataset is a function of the CATE and bias LS...
est_set = fread(paste0(OUTDIR,'LORD3_inputs_and_CATE_bias_Y.csv'))
# ... but the LORD3 LLR is just a function of seed1
LORD3_results = fread(paste0('output/simulation_1/',
							 seed1,'/LORD3_results.csv'))

df = cbind(est_set,LORD3_results)
# Order in decreasing order by LLR
df = df[order(-LLR)]
df$D = as.integer(df$D)

##########################################################
# Get CATE estimates at discovered RDs
##########################################################

print("Running Voroni KNN repair algorithm...")
L_x = as.matrix(df[1:M_prime,.(X1,X2)])
L_v = as.matrix(df[1:M_prime,nvec_col_names,with=F])
voroni_assets = get_voroni_knn_discontinuities_index_sets_and_estimates(
	L_x, L_v, df, k_prime, t_partition, estimator
)

top_M_prime_TEs = voroni_assets$top_M_prime_TEs

##########################################################
# Fit causal forest and get test set predictions
##########################################################

print("Fitting Causal Forest...")

Y.hat = predict(regression_forest(est_set[,.(X1,X2)], est_set[,Y]))$predictions
W.hat = predict(regression_forest(est_set[,.(X1,X2)], est_set[,D]))$predictions

params = tune_causal_forest(est_set[,.(X1,X2)], est_set[,Y], est_set[,D], Y.hat, W.hat)$params
print(params)

forest = causal_forest(est_set[,.(X1,X2)], est_set[,Y], est_set[,D],
  num.trees = 1000,
  min.node.size = as.numeric(params["min.node.size"]),
  sample.fraction = as.numeric(params["sample.fraction"]),
  mtry = as.numeric(params["mtry"]),
  alpha = as.numeric(params["alpha"]),
  imbalance.penalty = as.numeric(params["imbalance.penalty"])
)

test_X = fread('output/test_grid.csv')
cf_CATE_est = predict(forest,test_X)$prediction

fwrite(data.table(cf_CATE_est),paste0(OUTDIR,'test_grid_tau_hats_from_CF.csv'))

##########################################################
# Get causal forest estimates + bias estimates at centers
##########################################################

obs_est = predict(forest, top_M_prime_TEs[,.(X1,X2)])$predictions

true_CATE_and_e_bias = fread(paste0(OUTDIR,'true_CATE_and_bias_at_voroni_KNN_centers.csv'))
colnames(true_CATE_and_e_bias) = paste0('true_',colnames(true_CATE_and_e_bias
))

top_M_prime_TEs$obs_est = obs_est
top_M_prime_TEs = cbind(top_M_prime_TEs,true_CATE_and_e_bias)

top_M_prime_TEs$bias = top_M_prime_TEs$obs_est - top_M_prime_TEs$CATE

fwrite(top_M_prime_TEs,paste0(OUTDIR,'TE_bias_estimates_at_voroni_KNN_centers.csv'))

##########################################################
# Get causal forest estimates + bias estimates at each
# original data instance in df
##########################################################

obs_est = predict(forest, df[,.(X1,X2)])$predictions
df$obs_est = obs_est
setnames(df,c('CATE','bias'),c('true_CATE','true_bias'))
fwrite(df,paste0(OUTDIR,'obs_estimates_at_all_original_instances.csv'))

