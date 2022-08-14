
#####################################################################
## Load packages and utility functions
#####################################################################

library(rdrobust)
library(nprobust)
library(rdmulti)
library(data.table)
library(FNN)

make_threshold_assignments = function(data, latent_access){
	# Assign each unit an effective threshold
	# Two strategies: 
	#   1. `latent_access`: Give Cattaneo access to the latent complier type `G`
	#   2. Not `latent_access`: Partition based on observed `D`
	
	C = rep(NA,nrow(data))
	
	if (latent_access){
		# Assign G = C1 and C2 to l and h, respectively
		C[data$G == 'C1'] = 0.25
		C[data$G == 'C2'] = 0.65
		# Then randomly assign
		C[is.na(C)] = sample(c(0.25,0.65),sum(is.na(C)),replace=T)
	} else {
		# Randomly assign below low threshold and above high threshold.
		# In intermediate assign all untreated to l and all treated to h
		low_region = (data$X1	 < 0.25) | (data$X2 < 0.25)
		high_region = (data$X1  > 0.65) & (data$X2 > 0.65)
		mid_region = (!low_region) & (!high_region)
		
		C[low_region] = sample(c(0.25,0.65),sum(low_region),replace=T)
		C[high_region] = sample(c(0.25,0.65),sum(high_region),replace=T)
		C[mid_region & (data$D == 0)] = 0.25
		C[mid_region & (data$D == 1)] = 0.65
	}
	
	return(C)
}

setup_data = function(data,RD,latent_access=F){
	# Subset to the units exposed to the RD segments associated with orientation `RD`
	if (RD == 'vertical'){
		effective_data = data[(X1 < 0.25) |
													((X1 < 0.65) & (X2 > 0.25)) |
													((X2 > 0.65))]
		
		X = effective_data$X1
	} else if (RD == 'horizontal') {
		
		effective_data = data[(X2 < 0.25) |
													((X2 < 0.65) & (X1 > 0.25)) |
													((X1 > 0.65))]
		
		X = effective_data$X2
	} else {
		stop('Invalid RD')
	}
		 
	Y = effective_data$Y
	D = as.numeric(effective_data$D)
	c0 = 0.65
	c1 = 0.25
	C = make_threshold_assignments(effective_data,latent_access)
	effective_data[,C:=C]
	return(list(effective_data = effective_data,
							X = X,
							Y = Y,
							C = C,
							D = D,
							c0 = c0,
							c1 = c1))
}

get_extrapolated_TE_estimate = function(X,Y,D,C,c0,c1,cc,latent_access){
	#####################################################################
	## Extrapolation: c0 to cc
	#####################################################################
	
	# Low-cutoff group: estimates at c0
	
	aux1 = lprobust(Y[C==c0 & D==0],X[C==c0 & D==0],eval=c0)
	h00 = aux1$Estimate[2]
	N00 = aux1$Estimate[4]
	m00 = aux1$Estimate[5]
	m00.bc = aux1$Estimate[6]
	V00 = aux1$Estimate[8]^2
	
	
	# Low-cutoff group: estimates at cc
	
	aux2 = lprobust(Y[C==c0 & D==1],X[C==c0 & D==1],eval=cc)
	h01 = aux2$Estimate[2]
	N01 = aux2$Estimate[4]
	m01 = aux2$Estimate[5]
	m01.bc = aux2$Estimate[6]
	V01 = aux2$Estimate[8]^2
	
	
	# High-cutoff group: estimates at c0 and cc (with covariance)
	
	aux3 = lprobust(Y[C==c1 & D==0],X[C==c1 & D==0],eval=c(c0,cc),covgrid=TRUE,bwselect='mse-dpi')
	h10 = aux3$Estimate[1,2]
	N10 = aux3$Estimate[1,4]
	m10 = aux3$Estimate[1,5]
	m10.bc = aux3$Estimate[1,6]
	V10 = aux3$Estimate[1,8]^2
	h11 = aux3$Estimate[2,2]
	N11 = aux3$Estimate[2,4]
	m11 = aux3$Estimate[2,5]
	m11.bc = aux3$Estimate[2,6]
	V11 = aux3$Estimate[2,8]^2
	cov.rb = aux3$cov.rb[1,2]
	
	# Denominator: E[D | X = cc, C = l] (probability of treatment at cc for low group)
	if (latent_access){
		aux4 = lprobust(D[C==c0],X[C==c0],eval=c(cc))
		ED01 = aux4$Estimate[1,5]
		ED01.bc = aux4$Estimate[1,6]
	} else {
		ED01 = 1
		ED01.bc = 1
	}
	
	# TE: extrapolation Estimates using Theorem 2 Fuzzy RD estimator
	
	B = m00 - m10                                # Bias at low cutoff
	Dif = m01- m11                               # Naive difference at evaluation point
	TE = (Dif - B)/ED01                                 # Extrapolated TE estimate 
	TE.bc = (m01.bc - m11.bc - m00.bc + m10.bc)/ED01.bc    # Extrapolated TE RBC estimate 
	
	return(list(TE=TE,TE.bc=TE.bc))
}

get_extrapolated_estimates_at_ccs = function(data,ccs,orientation,latent_access=F){
	
	rd_segment_data = setup_data(data,orientation,latent_access)
	
	Y = rd_segment_data$Y
	X = rd_segment_data$X
	C = rd_segment_data$C
	D = rd_segment_data$D
	c0 = rd_segment_data$c0
	c1 = rd_segment_data$c1
	
	CATE_ests = rep(NA,length(ccs))
	i = 1
	for (cc in ccs){
		
		TE_ests = get_extrapolated_TE_estimate(X,Y,D,C,c0,c1,cc=cc,latent_access)
		CATE_ests[i] = TE_ests$TE.bc
		i = i+1
	}
	return(CATE_ests)
}


get_doubly_extrapolated_MSE = function(data,true_cates,ccs){
	horizontal_estimates = get_extrapolated_estimates_at_ccs(data,ccs,'horizontal')
	vertical_estimates = get_extrapolated_estimates_at_ccs(data,ccs,'vertical')
	
	
	grid_ests = data.frame(
		'X1' = rep(ccs,length(ccs)),
		'X2' = rep(ccs,each=length(ccs)),
		'X1_est' = rep(horizontal_estimates,length(ccs)),
		'X2_ests' = rep(vertical_estimates,each=length(ccs))
	)
	grid_ests$avg_est = (grid_ests$X1_est + grid_ests$X2_ests)/2
	
	
	NN = get.knnx(grid_ests[,c('X1','X2')],true_cates[,.(X1,X2)],k=1)$nn.index[,1]
	doubly_extrapolated_est = grid_ests[NN,'avg_est']
	true_cates$doubly_extrapolated_est = doubly_extrapolated_est
	
	doubly_extrapolated_MSE = mean((true_cates$doubly_extrapolated_est - true_cates$CATE)**2)
	print(paste('    MSE:',doubly_extrapolated_MSE))
	return(list(MSE=doubly_extrapolated_MSE,true_cates=true_cates))
}

#####################################################################
## Main 
#####################################################################

args = commandArgs(trailingOnly = TRUE)
seed1 = as.integer(args[1])
CATE_ls = args[2]
bias_ls = args[3]


OUTDIR = paste0('output/simulation_1/',seed1,'/',CATE_ls,'/',bias_ls,'/')

data = fread(paste0(OUTDIR,'/LORD3_inputs_and_CATE_bias_Y.csv'))
test_grid = fread('output/test_grid.csv')
true_cates = fread(paste0(OUTDIR,'/','true_test_CATE_and_bias.csv'))
true_cates = cbind(test_grid,true_cates)

ccs = unique(true_cates$X1)
ccs = ccs[(ccs>=0.25) & (ccs<=0.65)]
ccs = ccs[seq(1,length(ccs),2)]

MSE_and_updated_true_cates = get_doubly_extrapolated_MSE(data,true_cates,ccs)

true_cates = MSE_and_updated_true_cates$true_cates
MSE = MSE_and_updated_true_cates$MSE

to_write = data.frame('Variable'='MSE','Value'=MSE)
fwrite(to_write,paste0(OUTDIR,'/cattaneo_benchmark_description.csv'))