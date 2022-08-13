library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)

options(dplyr.summarise.inform = FALSE)
source('utils/voroni_knn.R')

M_prime = 400
k_prime = 1000
t_partition = 30
estimator = 'rotated_2SLS'
nvec_col_names = c('raw_nvec1','raw_nvec2')

################################################################
# First step: run LORD3 and Voroni KNN repair alg.
# Note get_voroni_knn_discontinuities_index_sets_and_estimates
# returns CATE estimates. These are an artifact for this run, 
# since Y is only sampled in step 3 of this sampling process.   
################################################################

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])

est_set = fread(paste0('output/simulation_1/',seed1,'/LORD3_inputs.csv'))
LORD3_results = fread(paste0('output/simulation_1/',seed1,'/LORD3_results.csv'))

# Bind and order in decreasing order by LLR
df = cbind(est_set,LORD3_results)
df = df[order(-LLR)]
df$D = as.integer(df$D)

print("Running Voroni KNN repair algorithm...")
L_x = as.matrix(df[1:M_prime,.(X1,X2)])
L_v = as.matrix(df[1:M_prime,nvec_col_names,with=F])
voroni_assets = get_voroni_knn_discontinuities_index_sets_and_estimates(
	L_x, L_v, df, k_prime, t_partition, estimator
)

top_M_prime_TEs = voroni_assets$top_M_prime_TEs

fwrite(top_M_prime_TEs,paste0('output/simulation_1/',seed1,'/voroni_KNN_centers__ignore_TE_estimates.csv'))