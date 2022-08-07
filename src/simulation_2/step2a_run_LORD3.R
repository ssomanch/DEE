library(data.table)
library(FNN)
library(IVcorrectr)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)

options(dplyr.summarise.inform = FALSE)

degree = 4
k = 200
nvec_col_names = c('raw_nvec1','raw_nvec2')

##########################################################
# Utilities
##########################################################

run_LORD3_and_cbind_useful_output = function(X,Y,D,degree,k,nvec_col_names){
	
	# Run LORD3
	lord3_results = LORD3_binary(X,Y,D,degree,k)

	# Extract likelihood ratio, rename normal vector columns, bind into data.table
	LLR = lord3_results$LRR
	normal_vectors = lord3_results$normal_vectors
	colnames(normal_vectors) = nvec_col_names
	all_results = as.data.frame(cbind(LLR,normal_vectors))
	setDT(all_results)
	return(all_results)
}

################################################################
# Run LORD3  
################################################################

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]

est_set = fread(paste0('../output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls,'/LORD3_inputs.csv'))

X = as.matrix(est_set[,.(X1,X2)])
dimnames(X) = NULL
Y = rnorm(nrow(X))
D = as.integer(est_set$D)

print("Running full sample LORD3...")
LORD3_results = run_LORD3_and_cbind_useful_output(X,Y,D,degree,k,nvec_col_names)

fwrite(LORD3_results,paste0('../output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls,'/LORD3_results.csv'))