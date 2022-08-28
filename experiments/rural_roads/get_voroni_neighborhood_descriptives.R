source('../LORD3_experiments/neighborhood_and_index_set_selection_utils.R')
source('src/balance_check_utils.R')

library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)
library(grf)
library(lsa)



########################################################

COS_POP_THRESH = 0.75
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

# Only run if y_name == transport_index_andrsn, since VKNN doesn't touch y
if (y_name != 'transport_index_andrsn'){
  stop('Only running for transport index...')
}
                                   
est_set = fread('output/rural_roads/roads_LORD3_w_RDD_category_and_p_val.csv')
raw_roads = fread('output/rural_roads/roads.csv')
scaled = scale(as.matrix(raw_roads[,.(longitude,latitude,total_pop)]))

if (RD_type == 'space_and_population'){
  OUTDIR = paste0('output/rural_roads/',y_name,'/space_and_population/',estimator,'__k_',k_prime,'__t_',t_partition,'__isotropic/')
} else {
  OUTDIR = paste0('output/rural_roads/',y_name,'/population/',estimator,'__k_',k_prime,'__t_',t_partition,'__isotropic/')
}
             

########################################################

get_pop_RD_flag = function(TE_df){
  nvec = TE_df[,c("longitude_nvec","latitude_nvec","total_pop_nvec")]
  nvec = as.matrix(nvec)
  nvec = nvec / sqrt(rowSums(nvec^2))
  target_v = c(0,0,1)
  abs_cos_sim = abs(apply(nvec,1,function(r) cosine(r,target_v)))
  pop_RDD = abs_cos_sim > COS_POP_THRESH
  return(pop_RDD)
}
                            
get_ball_sizes_ix = function(df,neighbors_and_sides,ix){
  neighs = df[neighbors_and_sides[[ix]][,'voroni_neighbors'],.(longitude_raw,latitude_raw,total_pop_raw)]
  mins = apply(neighs,2,min)
  maxs = apply(neighs,2,max)
  bw = maxs - mins
  return(as.data.frame(t(bw/2)))
}

get_all_bws = function(df,voroni_assets,pop_RDD){
  effective_BWs = rbindlist(lapply(1:length(voroni_assets$neighbors_and_sides),
                             function(ix) get_ball_sizes_ix(df,voroni_assets$neighbors_and_sides,ix)))
  effective_BWs[,pop_RDD:=ifelse(pop_RDD,'Population','Spatial')]
  
  another = copy(effective_BWs)
  another[,pop_RDD:='All RDs']
  doubled = rbind(effective_BWs,another)
  
  melted = melt(doubled,id.vars='pop_RDD')
  print(melted)
  aggs = melted %>% group_by(pop_RDD,variable) %>% 
          summarise(Mean=mean(value),Min=min(value),Max=max(value)) %>%
          mutate(Dimension=plyr::mapvalues(variable,
                                           c('longitude_raw','latitude_raw','total_pop_raw'),
                                           c('Longitude','Latitude','Population')),
                 `RD type`=pop_RDD)
  
  
  return(aggs[,c('RD type','Dimension','Mean','Min','Max')])
}
                                   
##########################################################
# Run VKNN
##########################################################

# Order by LLR and valid_RD, add top M_prime flag
df = est_set
setDT(df)
setnames(df,c('r2012',y_name),c('D','Y'))

unscaled = t(apply(df[,.(longitude,latitude,total_pop)],
                   1, function(r)r*attr(scaled,'scaled:scale') + attr(scaled, 'scaled:center')))
colnames(unscaled) = paste0(colnames(unscaled),'_raw')
df = cbind(df,unscaled)

# Sort first by LLR
df = df[order(-LLR),]
df[,top_M_prime:=c(rep(T,M_prime),rep(F,nrow(df)-M_prime))]

# Get first row
L = df[top_M_prime==T,]
L_r1 = c(`All RDs`=nrow(L),
         Spatial=nrow(L)-sum(L$total_pop_RD),Population=sum(L$total_pop_RD))

# Get second row
validated_L = L[valid_RD==T,]
validated_L_r2 = c(`All RDs`=nrow(validated_L),
                   Spatial=nrow(validated_L)-sum(validated_L$total_pop_RD),
                   Population=sum(validated_L$total_pop_RD))

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

pop_RDD = get_pop_RD_flag(voroni_assets$top_M_prime_TEs)
voroni_assets$top_M_prime_TEs[,total_pop_RD:=pop_RDD]

# Get third row
U = voroni_assets$top_M_prime_TEs
U_r3 = c(`All RDs`=nrow(U),
         Spatial=nrow(U)-sum(U$total_pop_RD),
         Population=sum(U$total_pop_RD))

print('Running balance check on VKNN RDDs...')

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

# Get fourth row
validated_U = voroni_assets$top_M_prime_TEs
validated_U_r4 = c(`All RDs`=nrow(validated_U),
         Spatial=nrow(validated_U)-sum(validated_U$total_pop_RD),
         Population=sum(validated_U$total_pop_RD))


full_size_df = data.frame(`$\\mathcal{L}$`=L_r1,`Validated $\\mathcal{L}$`=validated_L_r2,
                          `$\\mathcal{U}$`=U_r3,`Validated $\\mathcal{U}$`=validated_U_r4,
                          check.names = F)
sets = names(full_size_df)
full_size_df = as.data.frame(t(full_size_df))
                   
##########################################################
# Get descriptives
##########################################################
                   
# Get N villages in index sets by type
N_villages = sapply(voroni_assets$neighbors_and_sides,nrow)
N_villages_df = data.frame(N=N_villages,type=ifelse(voroni_assets$top_M_prime_TEs$total_pop_RD,'Population','Spatial'))
c2 = N_villages_df
c2$type = 'All RDs'
N_villages_df = rbind(N_villages_df,c2) %>% 
                  group_by(type) %>% 
                  summarise(Mean=round(mean(N)),Min=min(N),Max=max(N)) %>%
                  mutate(`Mean (min,max) villages in VK[D]`=paste0(Mean,' (',Min,'-',Max,')'))
N_row = pivot_wider(N_villages_df[,c('type','Mean (min,max) villages in VK[D]')],
                    names_from=type,values_from=`Mean (min,max) villages in VK[D]`)
N_row$Variable='Mean N (min,max)'

# Get bandwidths
all_bws = get_all_bws(df,voroni_assets,voroni_assets$top_M_prime_TEs$total_pop_RD) %>%
            mutate_if(is.numeric,round,digits=2)
mean_bw = pivot_wider(all_bws,id_cols=Dimension,values_from=Mean,names_from=`RD type`)
mean_bw = mean_bw %>% 
            rename(Variable=Dimension) %>% 
            mutate(Variable=plyr::mapvalues(Variable,c('Population','Longitude','Latitude'),c('Pop','Long','Lat'))) %>% 
            mutate(Variable = paste0(Variable,' BW'))
setDT(mean_bw)

# Stack for table 2
panel2 = rbind(mean_bw,N_row,fill=T)

fwrite(full_size_df,paste0(OUTDIR,'all_village_Ns.csv'),row.names=T)
fwrite(panel2,paste0(OUTDIR,'all_RD_bws.csv'))