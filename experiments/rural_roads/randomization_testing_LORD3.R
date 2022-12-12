library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)
library(foreach)
library(doParallel)
library(LORD3)

degree = 4
k = 200


args = commandArgs(trailingOnly=T)

#Number of randomizations
Q = as.integer(args[1])
root_path = args[2]
if (is.na(root_path)){
    root_path = ''
}

# set up combine function
#split = detectCores()
#cl = makeCluster(split)
cl = 128
registerDoParallel(cores=cl)

nvec_col_names = c('longitude_nvec','latitude_nvec','total_pop_nvec')

################################################################
# Run LORD3
################################################################

est_set = fread(paste0(root_path,'output/rural_roads/roads.csv'))
X = scale(as.matrix(est_set[,.(longitude,latitude,total_pop)]))
dimnames(X) = NULL
Y = est_set$transport_index_andrsn

max_LLRs = c()
for (q in 1:Q)
{
    # Randomize (permute) the treatment vector 
    D = sample(est_set$r2012)

    print(paste0("Running full sample LORD3 for randomization, round ", q))

    LORD3_results = run_LORD3_and_cbind_useful_output_parallel(X,Y,D,degree,k,nvec_col_names)

    max_LLRs <- append(max_LLRs, max(LORD3_results$LLR, na.rm=T))
}

write.csv(max_LLRs, 
            paste0(root_path,
                    'output/rural_roads/roads_LORD3_randomization_max_LLRs.csv'), 
                    row.names=F)