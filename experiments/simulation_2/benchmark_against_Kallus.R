

library(grf)
library(data.table)
library(FNN)

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]

OUTDIR = paste0('output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls)

M = 400

######################################################
# Set up the data                                    #
######################################################


data = fread(paste0(OUTDIR,'/LORD3_inputs.csv'))

first_N_rows = as.integer(nrow(data)/2)

# Use half the data as the confounded set
obs_set = data[1:first_N_rows,]

# Use subset, of the other half, that falls near the discovered RDs
# as the "unconfounded" set. Note this just highlights that this
# approach fails, since Y(0), Y(1) \not \perp T | X along the
# discontinuity (e.g. treatment isn't actually randomized at the
# RD -- since their is still selection into treatment at the RD --
# just not on the basis of the TE).
llrs = fread(paste0(OUTDIR,'/LORD3_results.csv'))
top_M = order(-llrs$LLR)[1:M]

knn = get.knn(data[,.(X1,X2)],k=200)

local_RD_est_set = unique(c(knn$nn.index[top_M,]))
local_RD_est_set = setdiff(local_RD_est_set,1:first_N_rows)

local_RD_est_set = data[local_RD_est_set,]

# Load the test points
test_data = fread('../../GPCorrection/output/test_data_for_R.csv')
true_cates = fread(paste0(OUTDIR,'/true_test_CATE_and_bias.csv'))

######################################################
# Fit the observational estimator over half the data #
######################################################

print("Fitting Causal Forest...")

Y.hat = predict(regression_forest(obs_set[,.(X1,X2)], obs_set[,Y]))$predictions
W.hat = predict(regression_forest(obs_set[,.(X1,X2)], obs_set[,D]))$predictions

params = tune_causal_forest(obs_set[,.(X1,X2)], obs_set[,Y], obs_set[,D], Y.hat, W.hat)$params

forest = causal_forest(obs_set[,.(X1,X2)], obs_set[,Y], obs_set[,D],
  num.trees = 1000,
  min.node.size = as.numeric(params["min.node.size"]),
  sample.fraction = as.numeric(params["sample.fraction"]),
  mtry = as.numeric(params["mtry"]),
  alpha = as.numeric(params["alpha"]),
  imbalance.penalty = as.numeric(params["imbalance.penalty"])
)

######################################################
# Find the "unconfounded" set (NOT UNCONFOUNDED) and #
# fit a linear model on the bias target              #
######################################################

print("Fitting debiasing linear model...")

# First fit propensity model and get reweighted target
tuned_q_params = tune_regression_forest(local_RD_est_set[,.(X1,X2)], local_RD_est_set[,D])$params
q = regression_forest(obs_set[,.(X1,X2)], obs_set[,D],
  num.trees = 1000,
  min.node.size = as.numeric(tuned_q_params["min.node.size"]),
  sample.fraction = as.numeric(tuned_q_params["sample.fraction"]),
  mtry = as.numeric(tuned_q_params["mtry"]),
  alpha = as.numeric(tuned_q_params["alpha"]),
  imbalance.penalty = as.numeric(tuned_q_params["imbalance.penalty"])
)
eD = predict(q,local_RD_est_set[,.(X1,X2)])$predictions
q = (local_RD_est_set$D/eD) - (1 - local_RD_est_set$D)/(1-eD)

# Then get confounded estimate at points in "unconfounded" set
omega = predict(forest,local_RD_est_set[,.(X1,X2)])$predictions
kallus_target = q*local_RD_est_set$Y - omega

# Fit linear debiasing model
callus_model_df = data.frame('X1'=local_RD_est_set$X1,
                             'X2'=local_RD_est_set$X2,
                             'target'=kallus_target)
callus_model = lm('target ~ X1 + X2', callus_model_df)

######################################################
# Load the test data              #
######################################################

callus_pred = predict(callus_model,test_data)
omega_full = predict(forest,test_data[,.(X1,X2)])$predictions
tau_full = callus_pred + omega_full

MSE = mean((tau_full - true_cates$CATE)**2)
print(paste('    MSE:',MSE))
to_write = data.frame('Variable'='MSE','Value'=MSE)
fwrite(to_write,paste0(OUTDIR,'/kallus_benchmark_description.csv'))