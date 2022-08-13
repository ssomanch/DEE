
library(data.table)
library(FNN)
library(LORD3)
library(AER)

degree = 4
k = 200
M_prime = 400
estimator = 'nonparametric_estimator'
nvec_col_names = c('raw_nvec1','raw_nvec2')

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]



get_top_M_CATE_ests = function(df,knn_assignments,M_prime,estimator){
	# Get the local 2SLS estimates for the first M_prime rows in df
    # Note we assume that df is in descending order by LLR.

	CATEs = rep(NA,M_prime)
	ses = rep(NA,M_prime)
	dfs = rep(NA,M_prime)
	
	# Compute side assignments
	for (i in 1:M_prime){
		cx = unlist(df[i,.(X1,X2)])
		v = unlist(df[i,.(raw_nvec1,raw_nvec2)])
	
		# Find the index of the neighbors
		n_ix = knn_assignments[i,]
	
		# Partition across RD hyperplane
		nX = as.matrix(df[n_ix,.(X1,X2)])
		cXs = center_neighborhood_ball(nX,1:nrow(nX),cx)
		S = bisect_neighborhood(cXs,v)
	
		# Get neighbors Y and T
		ny = df[n_ix,Y]
		nt = df[n_ix,D]
		
		# Compute estimate using this k-neighborhood
		est_f = get(estimator)
		estimate = est_f(S,ny,nt,cXs,v)
		CATEs[i] = estimate$tau
		ses[i] = estimate$se
		dfs[i] = estimate$df
	}
	
	CATE_ests = cbind(CATE = CATEs,
									  ses = ses,
									  lower = CATEs + qt(0.025,df=dfs)*ses,
									  upper = CATEs + qt(0.975,df=dfs)*ses,
									  df = dfs)
	return(CATE_ests)
}

##########################################################
# Load data
##########################################################

OUTDIR = paste0('output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls,'/')

# The dataset is a function of the CATE and bias LS...
df = fread(paste0(OUTDIR,'obs_estimates_at_all_original_instances.csv'))

# Order in decreasing order by LLR
df = df[order(-LLR)]
df$D = as.integer(df$D)

##########################################################
# Get the KNN ball assignments
##########################################################
knn_assignments = get.knn(df[,.(X1,X2)],k=k)$nn.index

##########################################################
# Compute the local 2SLS estimates and bias in CF
##########################################################
CATE_ests = get_top_M_CATE_ests(df,knn_assignments,M_prime,estimator)

top_M_prime = df[1:M_prime,.(X1,X2,raw_nvec1,raw_nvec2,true_CATE,true_bias,obs_est)]
bound = cbind(top_M_prime,CATE_ests)
bound$bias = bound$obs_est - bound$CATE

fwrite(bound,paste0(OUTDIR,'unfiltered_estimates_in_L.csv'))