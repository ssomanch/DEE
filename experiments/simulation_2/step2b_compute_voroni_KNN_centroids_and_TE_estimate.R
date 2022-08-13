library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)

options(dplyr.summarise.inform = FALSE)
source('../neighborhood_and_index_set_selection_utils.R')

M_prime = 400
k_prime = 400
t_partition = 30
estimator = 'nonparametric_estimator'
nvec_col_names = c('raw_nvec1','raw_nvec2')

################################################################
# Run Voroni KNN algorithm, and compute CATE estimates.
# Note in this simulation, where treatment and potential
# outcomes are not independent, we sample Y(1) and Y(0)
# in the first step of the simulation. Thus, the CATE
# estimates we obtain here are not just placeholders on
# noise Ys, but the actual CATE estimates.    
################################################################

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]

OUTDIR = paste0('../output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls,'/')

est_set = fread(paste0(OUTDIR,'LORD3_inputs.csv'))
LORD3_results = fread(paste0(OUTDIR,'LORD3_results.csv'))

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

fwrite(top_M_prime_TEs,paste0(OUTDIR,'TE_estimates_at_voroni_KNN_centers_without_true_parameters.csv'))
