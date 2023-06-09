library(data.table)
library(FNN)
library(LORD3)
library(grf)
library(dplyr)
library(tidyr)
library(mvtnorm)
library(nakagami)
library(Matrix)
library(AER)
library(pracma)

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]

##########################################################
# Load data
##########################################################

OUTDIR = paste0('output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls,'/')

est_set = fread(paste0(OUTDIR,'/LORD3_inputs.csv'))
LORD3_results = fread(paste0(OUTDIR,'/LORD3_results.csv'))

df = cbind(est_set,LORD3_results)
# Order in decreasing order by LLR
df = df[order(-LLR)]
df$D = as.integer(df$D)

top_M_prime_TEs = fread(paste0(OUTDIR,'/TE_estimates_at_voroni_KNN_centers_without_true_parameters.csv'))

##########################################################
# Fit causal forest and get test set predictions
##########################################################

forest = causal_forest(est_set[,.(X1,X2)], est_set[,Y], est_set[,D],
  num.trees = 1000,
  tune.parameters='all'
)

test_X = fread('output/test_grid.csv')
cf_CATE_est = predict(forest,test_X)$prediction

fwrite(data.table(cf_CATE_est),paste0(OUTDIR,'test_grid_tau_hats_from_CF.csv'))

##########################################################
# Get causal forest estimates + bias estimates at centers
##########################################################

obs_est = predict(forest, top_M_prime_TEs[,.(X1,X2)])$predictions

true_CATE_and_e_bias = fread(paste0(OUTDIR,'true_CATE_and_bias_at_voroni_KNN_centers.csv'))
setnames(true_CATE_and_e_bias,c('CATE','bias'),c('true_CATE','true_bias'))

top_M_prime_TEs$obs_est = obs_est
top_M_prime_TEs = cbind(top_M_prime_TEs,true_CATE_and_e_bias)

top_M_prime_TEs$bias = top_M_prime_TEs$obs_est - top_M_prime_TEs$CATE

fwrite(top_M_prime_TEs,paste0(OUTDIR,'TE_bias_estimates_at_voroni_KNN_centers_with_true_parameters.csv'))

##########################################################
# Get causal forest estimates + bias estimates at each
# original data instance in df
##########################################################

obs_est = predict(forest, df[,.(X1,X2)])$predictions
df$obs_est = obs_est

setnames(df,c('tau','bias'),c('true_CATE','true_bias'))
fwrite(df,paste0(OUTDIR,'obs_estimates_at_all_original_instances.csv'))
