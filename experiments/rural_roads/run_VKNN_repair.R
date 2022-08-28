library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)
library(grf)

source('experiments/rural_roads/balance_check_utils.R')
source('utils/voroni_knn.R')

########################################################
alpha = 0.05
M_prime = 1500
nvec_col_names = c('longitude_nvec','latitude_nvec','total_pop_nvec')
control_cols = c('primary_school','med_center','elect',
                 'tdist','irr_share','ln_land','pc01_lit_share',
                 'pc01_sc_share','bpl_landed_share',
                 'bpl_inc_source_sub_share','bpl_inc_250plus')

alpha = alpha/length(control_cols)

# Parse target Y
args = commandArgs(trailingOnly=TRUE)
y_name = args[1]
RD_type = args[2]
estimator = args[3]
k_prime = as.integer(args[4])
t_partition = as.integer(args[5])

if (RD_type == 'space_and_population'){
  OUTDIR = paste0('output/rural_roads/',y_name,'/space_and_population/',estimator,'__k_',k_prime,'__t_',t_partition,'__isotropic/')
} else {
  OUTDIR = paste0('output/rural_roads/',y_name,'/population/',estimator,'__k_',k_prime,'__t_',t_partition,'__isotropic/')
}
dir.create(OUTDIR, showWarnings = FALSE, recursive = T)
est_set = fread('output/rural_roads/roads_LORD3_w_RDD_category_and_p_val.csv')
raw_roads = fread('output/rural_roads/roads.csv')
scaled = scale(as.matrix(raw_roads[,.(longitude,latitude,total_pop)]))

##########################################################
# Load causal forest
##########################################################

forest = readRDS(file = paste0('output/rural_roads/',y_name,'/','forest.rds'))

##########################################################
# Run VKKN and get obs_est
##########################################################

# Order by LLR and valid_RD, add top M_prime flag
df = est_set
setDT(df)
setnames(df,c('r2012',y_name),c('D','Y'))

# Sort first by LLR
df = df[order(-LLR),]
df[,top_M_prime:=c(rep(T,M_prime),rep(F,nrow(df)-M_prime))]

if (RD_type == 'population'){
  # Then sort valid population RDs to top. Will stable sort valid population RDs to top, then rest of RDs
  df = df[order(-(total_pop_RD & valid_RD)),]
  L_x = as.matrix(df[top_M_prime & total_pop_RD & valid_RD,.(longitude,latitude,total_pop)])
  L_v = as.matrix(df[top_M_prime & total_pop_RD & valid_RD,nvec_col_names,with=F])
} else {
  df = df[order(-valid_RD),]
  L_x = as.matrix(df[top_M_prime & valid_RD,.(longitude,latitude,total_pop)])
  L_v = as.matrix(df[top_M_prime & valid_RD,nvec_col_names,with=F])
}

print("Running Voroni KNN repair algorithm...")
voroni_assets = get_voroni_knn_discontinuities_index_sets_and_estimates(
  L_x, L_v, df, k_prime, t_partition, estimator
)

print('Running balance check on VKNN RDDs...')
print(paste('Initial number of VKNN neighborhoods:',nrow(voroni_assets$top_M_prime_TEs)))

setnames(df,c('D'),c('r2012'))
Xs = as.matrix(df[,.(longitude,latitude,total_pop)])
all_balance_voroni = lapply(
  1:nrow(voroni_assets$top_M_prime_TEs), function(ix) run_balance_one_VKNN_neighborhood(
    Xs,voroni_assets,df,ix,control_cols,run_balance_one_outcome
  )
)
all_balance_voroni_df = rbindlist(lapply(all_balance_voroni, as.data.frame.list))
min_P_VKNN = apply(all_balance_voroni_df,1,min)
valid_VKNN = min_P_VKNN > alpha

voroni_assets$top_M_prime_TEs = voroni_assets$top_M_prime_TEs[valid_VKNN,]
voroni_assets$neighbors_and_sides = voroni_assets$neighbors_and_sides[which(valid_VKNN)]

print(paste('Final number of VKNN neighborhoods:',nrow(voroni_assets$top_M_prime_TEs)))

top_M_prime_TEs = voroni_assets$top_M_prime_TEs
top_M_prime_TEs$obs_est = predict(forest,top_M_prime_TEs[,.(longitude,latitude)])$prediction

unscaled = t(apply(top_M_prime_TEs[,.(longitude,latitude,total_pop)],
                   1, function(r)r*attr(scaled,'scaled:scale') + attr(scaled, 'scaled:center')))
colnames(unscaled) = paste0(colnames(unscaled),'_raw')
top_M_prime_TEs = cbind(top_M_prime_TEs,unscaled)
                
fwrite(top_M_prime_TEs,paste0(OUTDIR,'roads_VKNN_ests.csv'))


