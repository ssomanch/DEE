source('../LORD3_experiments/neighborhood_and_index_set_selection_utils.R')

library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)
library(grf)


# Parse target Y
args = commandArgs(trailingOnly=TRUE)
y_name = args[1]

OUTDIR = paste0('output/rural_roads/',y_name,'/')

dir.create(OUTDIR, showWarnings = FALSE)
est_set = fread('output/rural_roads/roads_LORD3.csv')

##########################################################
# Fit causal forest
##########################################################

Y = est_set[[y_name]]
D = est_set$r2012

Y.hat = predict(regression_forest(est_set[,.(longitude,latitude)], Y))$predictions
W.hat = predict(regression_forest(est_set[,.(longitude,latitude)], D))$predictions

params = tune_causal_forest(est_set[,.(longitude,latitude)], Y, D, Y.hat, W.hat)$params
print(params)

forest = causal_forest(est_set[,.(longitude,latitude)], Y, D,
  num.trees = 1000,
  min.node.size = as.numeric(params["min.node.size"]),
  sample.fraction = as.numeric(params["sample.fraction"]),
  mtry = as.numeric(params["mtry"]),
  alpha = as.numeric(params["alpha"]),
  imbalance.penalty = as.numeric(params["imbalance.penalty"])
)

cf_CATE_est_w_var = predict(forest,est_set[,.(longitude,latitude)],
                            estimate.variance = TRUE)

est_set$obs_est = cf_CATE_est_w_var$prediction
est_set$obs_est_var = cf_CATE_est_w_var$variance.estimates
fwrite(est_set,paste0(OUTDIR,'roads_with_CATE.csv'))

saveRDS(forest,file=paste0(OUTDIR,'forest.rds'))

