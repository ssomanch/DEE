library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)
library(grf)
library(reshape2)
library(tidyr)
library(ggplot2)
library(stringr)

source('experiments/rural_roads/balance_check_utils.R')

variable_map = c(
  primary_school = 'Primary school',
  med_center = 'Medical center',
  elect = 'Electrified',
  tdist = 'Distance from nearest town (km)',
  irr_share = 'Land irrigated (share)',
  ln_land = 'In land area',
  pc01_lit_share = 'Literate (share)',
  pc01_sc_share = 'Scheduled caste (share)',
  bpl_landed_share = 'Land ownership (share)',
  bpl_inc_source_sub_share = 'Subsistence ag (share)',
  bpl_inc_250plus = 'HH income > INR 250 (share)'
)

# M_prime is chosen based on randomization testing by running 
# randomization_testing_LORD3.R and  compute_M_prime_from_randomization.R
# Note that these files are not run in the regular flow of the application.
M_prime = 2784
alpha = 0.05
nvec_col_names = c('longitude_nvec','latitude_nvec','total_pop_nvec')

##########################################################
# Run balance checks from Table 1 in original paper
##########################################################

# Load data
est_set = fread('output/rural_roads/roads_LORD3_w_RDD_category.csv')
controls = fread('output/rural_roads/roads_w_controls.csv')
control_cols = colnames(controls)[12:22]

alpha = alpha/length(control_cols)

bound = cbind(est_set,controls[,control_cols,with=F])
bound[,orig_order := 1:nrow(bound)]
               
# Order by LLR, add top M_prime flag
bound = bound[order(-LLR),]
bound[,top_M_prime:=c(rep(T,M_prime),rep(F,nrow(bound)-M_prime))]

# Get neighbors
Xs = as.matrix(bound[,.(longitude,latitude,total_pop)])
knns = get.knn(Xs,k=200)

# Run balance checks and compute the minimum p-value across each discovered RD
all_balance = lapply(
  1:M_prime, function(ix) run_balance_one_neighborhood(
    Xs,knns,bound,ix,control_cols,run_balance_one_outcome
  )
)
all_balance_df = rbindlist(lapply(all_balance, as.data.frame.list))

min_P = apply(all_balance_df,1,min)
valid_RD = min_P > alpha   
 
##########################################################
# Plot p-values across baseline covariates
##########################################################
               
all_balance_df[,PopulationRD:=bound[1:M_prime][['total_pop_RD']]]
all_balance_df[,valid_RD:=valid_RD]

melted = melt(all_balance_df,id_vars = 'PopulationRD')
melted$variable = plyr::mapvalues(melted$variable,names(variable_map),variable_map)
melted$variable = str_wrap(melted$variable, width = 20)
            
p1 = ggplot(data=melted,aes(x=value)) + geom_histogram() + 
  facet_grid(variable~PopulationRD,labeller = labeller(PopulationRD=label_both,variable=label_value)) +
  xlab('P-value for balance test') + ylab('Count (N discovered RDs in top M=1,500)') +
  ggtitle('Balance checks on discovered RDs')
ggsave('output/rural_roads/figures/balance_p_vals.png',plot=p1,height=15,width=5)
              
bound[,valid_RD:=c(valid_RD,rep(F,nrow(bound)-M_prime))]
bound[,min_P:=c(min_P,rep(0,nrow(bound)-M_prime))]
               
bound = bound[order(orig_order),]
fwrite(bound,'output/rural_roads/roads_LORD3_w_RDD_category_and_p_val.csv')

# Make a final plot showing min-p
bound[,`RD type`:=ifelse(total_pop_RD,'Population','Spatial')]
p2 = ggplot(data=bound[top_M_prime == TRUE,],aes(x=min_P)) + geom_histogram(binwidth=alpha) + 
  facet_grid(~`RD type`,labeller = label_both, scales='free_y') +
  xlab('min(p) for balance test') + ylab('Count (N discovered RDs in top M=1,500)') +
  ggtitle('Balance checks on discovered RDs') + 
  geom_vline(xintercept = alpha,linetype=2,color='red')
ggsave('output/rural_roads/figures/balance_min_p_vals.png',plot=p2,height=4,width=7)