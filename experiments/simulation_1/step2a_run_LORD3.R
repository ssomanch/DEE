library(data.table)
library(LORD3)
library(Matrix)

options(dplyr.summarise.inform = FALSE)

degree = 4
k = 200
nvec_col_names = c('raw_nvec1','raw_nvec2')

################################################################
# Run LORD3 and Voroni KNN repair alg.
# Note get_voroni_knn_discontinuities_index_sets_and_estimates
# returns CATE estimates. These are an artifact for this run, 
# since Y is only sampled in step 3 of this sampling process.   
################################################################

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])

est_set = fread(paste0('../output/simulation_1/',seed1,'/LORD3_inputs.csv'))

X = as.matrix(est_set[,.(X1,X2)])
dimnames(X) = NULL
Y = rnorm(nrow(X))
D = as.integer(est_set$D)

print("Running full sample LORD3...")
LORD3_results = run_LORD3_and_cbind_useful_output(X,Y,D,degree,k,nvec_col_names)

fwrite(LORD3_results,paste0('../output/simulation_1/',seed1,'/LORD3_results.csv'))