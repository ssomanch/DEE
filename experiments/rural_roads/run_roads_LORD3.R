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
nvec_col_names = c('longitude_nvec','latitude_nvec','total_pop_nvec')

################################################################
# Run LORD3
################################################################

est_set = fread('output/rural_roads/roads.csv')
X = scale(as.matrix(est_set[,.(longitude,latitude,total_pop)]))
dimnames(X) = NULL
Y = est_set$transport_index_andrsn
D = est_set$r2012

print("Running full sample LORD3...")

# set up combine function
#split = detectCores()
#cl = makeCluster(split)
cl = 50
registerDoParallel(cores=cl)

LORD3_results = run_LORD3_and_cbind_useful_output_parallel(X,Y,D,degree,k,nvec_col_names)

bound = data.frame(cbind(est_set,LORD3_results))
for (col in c('longitude','latitude','total_pop')){
  bound[col] = scale(bound[col])
}
fwrite(bound,'output/rural_roads/roads_LORD3.csv')