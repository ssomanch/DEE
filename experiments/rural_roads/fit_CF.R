source('utils/voroni_knn.R')

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

forest = causal_forest(est_set[,.(longitude,latitude)], Y, D,
  num.trees = 1000,
  tune.parameters='all'
)

cf_CATE_est_w_var = predict(forest,est_set[,.(longitude,latitude)],
                            estimate.variance = TRUE)

est_set$obs_est = cf_CATE_est_w_var$prediction
est_set$obs_est_var = cf_CATE_est_w_var$variance.estimates
fwrite(est_set,paste0(OUTDIR,'roads_with_CATE.csv'))

saveRDS(forest,file=paste0(OUTDIR,'forest.rds'))

