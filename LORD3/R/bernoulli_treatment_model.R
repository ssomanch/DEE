bernoulli_multiplicative_odds_score_function = function(t, p_hat_x, multiplier){
  # Score function e.g. derivative of ll w.r.t. multipler parameters
  return(sum(t/multiplier - p_hat_x/(1 - p_hat_x + multiplier*p_hat_x)))
}

binary_search_multiplier = function(t, p_hat_x,min_multiplier, max_multiplier,tol=1e-4){
  # Binary search given valid min/max bounds on multiplier
  middle_val = (max_multiplier + min_multiplier)/2
  grad_val = bernoulli_multiplicative_odds_score_function(t, p_hat_x,middle_val)
  while (abs(grad_val)>tol){
    if(grad_val>0){
      # Search right half since grad decreases with multiplier
      min_multiplier = middle_val
    } else {
      # Search left half
      max_multiplier = middle_val
    }
    middle_val = (min_multiplier + max_multiplier)/2 # New midpoint
    grad_val = bernoulli_multiplicative_odds_score_function(t, p_hat_x,middle_val)
  }
  return(middle_val)
}

multiplier_binary_search = function(t, p_hat_x, max_init=1,tol=1e-6){
  # If t is all 0/1 will fail.
  if((sum(t)!=length(t)) & (sum(1-t)!=length(t))){

    # Find closed interval for search
    max_multiplier = max_init
    min_multiplier = 0
    grad_val = bernoulli_multiplicative_odds_score_function(t, p_hat_x, max_multiplier)
    while(grad_val>0){
      min_multiplier = max_multiplier
      max_multiplier = max_multiplier*2
      grad_val = bernoulli_multiplicative_odds_score_function(t, p_hat_x, max_multiplier)
    }
    return(binary_search_multiplier(t, p_hat_x,min_multiplier, max_multiplier,tol))
  } else{
    return(NA)
  }
}

get_group_mu_binary = function(groups,nt,np_hat_x){
  # Compute the mu vector for the each point (e.g. each point assigned beta_g for its group)
  g0_beta_hat = multiplier_binary_search(nt[groups==0],np_hat_x[groups==0])
  g1_beta_hat = multiplier_binary_search(nt[groups==1],np_hat_x[groups==1])
  mu = (1 - groups)*g0_beta_hat + groups*g1_beta_hat
  return(mu)
}

get_group_LRR_binary = function(group_mu,nt,np_hat_x,beta_0_hat){
  # Get the group LRR given in equation (9)
  term1 = nt*log(group_mu/beta_0_hat)
  term2 = log(1 - np_hat_x + beta_0_hat*np_hat_x)
  term3 = -log(1 - np_hat_x + group_mu*np_hat_x)
  return(sum(term1 + term2 + term3))
}

get_all_group_LLRs = function(cX,nt,np_hat_x){
  #########################################
  # New faster version avoids recomputing #
  # mu and LLR for identical partitions   #
  #########################################
  cn_groups = all_groups(cX)

  # Compute full group (null) multiplier
  beta_0_hat = multiplier_binary_search(nt, np_hat_x)

  # Get equivalence matrix
  # Given a group assignment g1, we can identify if it
  # is equivalent to another assignment g2 by checking if
  #     - g2 is true everywhere g1 is true
  #     - g2 is false everywhere g2 is false
  n_true = colSums(cn_groups)
  true_where_true = t(cn_groups) %*% cn_groups

  false_where_false  = t(cn_groups) %*% (! cn_groups)

  # There are two cases: two groups are identifical if
  #   Case 1:
  #     true_where_true == n_true (e.g. g2 is true evenywhere g1 is true), &
  #     false_where_false == 0 (e.g. g2 is false everywhere g1 is false)
  #   Case 2:
  #     true_where_true == 0 (e.g. g2 is false everywhere g1 is true) &
  #     false_where_false == nrow(cn_groups) - n_true (e.g. g2 is true everwhere g1 is false)

  all_true_where_true = sweep(true_where_true,2,n_true,'==')
  all_true_where_false = sweep(true_where_true,2,nrow(cn_groups) - n_true,'==')

  can_impute = ((all_true_where_true & (false_where_false == 0)) +
                  ((true_where_true == 0) & (all_true_where_false)))

  # Replace upper triangle with 0s -- don't need to backfill
  can_impute[upper.tri(can_impute,diag=T)] = 0

  mu = matrix(NA,nrow=nrow(cX),ncol=nrow(cX))
  lrrs = rep(NA,ncol(cn_groups))
  for (j in 1:ncol(cn_groups)){

    # First check if we've already computed lrrrs
    if (is.na(lrrs[j])){
      mu[,j] = get_group_mu_binary(cn_groups[,j],nt,np_hat_x)
      lrrs[j] = get_group_LRR_binary(mu[,j],nt,np_hat_x,beta_0_hat)

      # Then fill any jaccard neighbors
      can_impute_for_j = which(can_impute[,j]>0)
      mu[,can_impute_for_j] = mu[,j]
      lrrs[can_impute_for_j] = lrrs[j]
    }
  }
  return(lrrs)
}

get_lrr_binary = function(cn, nt, np_hat_x){
  # cn_groups is a kxk matrix; the ith column gives the grouping
  # induced by the halfplane with normal vector given by the ith
  # vector in the group
  cn_groups = all_groups(cn)

  # Compute full group (null) multiplier
  beta_0_hat = multiplier_binary_search(nt, np_hat_x)

  # Compute mu_i for each group; a vector giving the group mean estimate
  mu = apply(cn_groups,2,function(group) get_group_mu_binary(group,nt,np_hat_x))

  # mu is a kxk matrix; each column gives group mean estimates for each point
  LRR = apply(mu,2,function(group_mu) get_group_LRR_binary(group_mu,nt,np_hat_x,beta_0_hat))
  return(LRR)
}

get_each_points_max_LRR_binary = function(X, neighbors, y, t, p_hat_x){

  estimators = list(nonparametric_estimator=nonparametric_estimator,
                    rotated_2SLS=rotated_2SLS)
  estimates = list()

  max_LRR = rep(0,dim(neighbors)[1])
  observed_prop_delta = rep(NA,dim(neighbors)[1])
  max_normal_vector = matrix(NA,dim(X)[1],dim(X)[2])

  for (ix in 1:dim(neighbors)[1]){
    neighborhood_ix = neighbors[ix,]
    cneighborhood = center_neighborhood_ball(X,neighborhood_ix,X[ix,])
    nt = t[neighborhood_ix]
    ny = y[neighborhood_ix]
    np_hat_x = p_hat_x[neighborhood_ix]
    lrrs = get_all_group_LLRs(cneighborhood, nt, np_hat_x)
    max_LRR[ix] = suppressWarnings(max(lrrs,na.rm=T)) # Throws warnings when all NA -- which happens if homogeneous groups
    if (!is.infinite(max_LRR[ix])){
      max_normal_vector[ix,] = cneighborhood[which.max(lrrs),]
      best_grouping = bisect_neighborhood(cneighborhood, max_normal_vector[ix,])

      # Get estimates
      ix_estimates = list()
      for (est_ix in 1:length(estimators)){
        estimator = estimators[[est_ix]]
        est_name = names(estimators)[est_ix]
        est = estimator(as.integer(best_grouping),ny,nt,cneighborhood,max_normal_vector[ix,])
        names(est) = paste0(paste0(est_name,'__'),names(est))
        ix_estimates = c(ix_estimates,est)
      }
      estimates[[ix]] = ix_estimates


      observed_prop_delta[ix] = mean(nt[which(best_grouping==1)]) - mean(nt[which(best_grouping==0)])
    } else {
      estimates[[ix]] = list(NA)
    }
  }

  # Flatten estimates into a data.table
  estimates = rbindlist(estimates,fill=T)

  return(list(LRR=max_LRR,normal_vectors=max_normal_vector,
              observed_prop_delta=observed_prop_delta,
              estimates=estimates))
}

LORD3_binary = function(X,binary_y,binary_t,degree,k){

  #### First we model T
  df = as.data.frame(cbind(X,binary_t))
  x_names = paste0('x',1:ncol(X))
  colnames(df) = c(x_names,'t')
  smooth_formula = as.formula(paste0('t~polym(',paste(x_names,collapse=', '),
                                     ', degree=',degree,', raw=T)'))
  binary_t_hat_func = glm(family=binomial(link='logit'),formula = smooth_formula, singular.ok=F, data=df)
  p_hat_x = predict(binary_t_hat_func, df, type = "response")

  # Find the neighborhoods
  binary_neighbors = get.knn(X, k = k)

  # Get full results
  result = get_each_points_max_LRR_binary(X,binary_neighbors$nn.index, binary_y, binary_t, p_hat_x)
  result[['t_hat_func']] = binary_t_hat_func
  return(result)
}


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


###################################
# Parallelizing #
###################################

scan_one_point = function(ix,X,neighbors,y,t,p_hat_x,estimators){
	neighborhood_ix = neighbors[ix,]
	cneighborhood = center_neighborhood_ball(X,neighborhood_ix,X[ix,])
	nt = t[neighborhood_ix]
	ny = y[neighborhood_ix]
	np_hat_x = p_hat_x[neighborhood_ix]
	lrrs = get_all_group_LLRs(cneighborhood, nt, np_hat_x)
	max_LLR = suppressWarnings(max(lrrs,na.rm=T)) # Throws warnings when all NA -- which happens if homogeneous groups
	if (!is.infinite(max_LLR)){
		max_normal_vector = cneighborhood[which.max(lrrs),]
		best_grouping = bisect_neighborhood(cneighborhood, max_normal_vector)
		
		# Get estimates
		ix_estimates = list()
		for (est_ix in 1:length(estimators)){
			estimator = estimators[[est_ix]]
			est_name = names(estimators)[est_ix]
			est = estimator(as.integer(best_grouping),ny,nt,cneighborhood,max_normal_vector)
			names(est) = paste0(paste0(est_name,'__'),names(est))
			ix_estimates = c(ix_estimates,est)
		}
		
		observed_prop_delta = mean(nt[which(best_grouping==1)]) - mean(nt[which(best_grouping==0)])
	} else {
		max_normal_vector = rep(NA,ncol(X))
		observed_prop_delta = NA
		ix_estimates = list()
		for (est_ix in 1:length(estimators)){
			est = list(df=NA,se=NA,tau=NA)
			est_name = names(estimators)[est_ix]
			names(est) = paste0(paste0(est_name,'__'),names(est))
			ix_estimates = c(ix_estimates,est)
		}
	}
	return (list(max_LLR=max_LLR,
							 max_normal_vector=max_normal_vector,
							 observed_prop_delta=observed_prop_delta,
							 ix_estimates=ix_estimates))
}


get_each_points_max_LRR_binary_parallel = function(X, neighbors, y, t, p_hat_x){
	
	estimators = list(nonparametric_estimator=nonparametric_estimator,
										rotated_2SLS=rotated_2SLS)
	
	
	each_point = foreach(ix=1:dim(neighbors)[1], .packages=c('AER','LORD3','data.table')) %dopar% {
		scan_one_point(ix,X,neighbors,y,t,p_hat_x,estimators)
	}
	return(each_point)
}


LORD3_binary_parallel = function(X,binary_y,binary_t,degree,k){
	
	#### First we model T
	df = as.data.frame(cbind(X,binary_t))
	x_names = paste0('x',1:ncol(X))
	colnames(df) = c(x_names,'t')
	smooth_formula = as.formula(paste0('t~polym(',paste(x_names,collapse=', '),
																		 ', degree=',degree,', raw=T)'))
	binary_t_hat_func = glm(family=binomial(link='logit'),formula = smooth_formula, singular.ok=F, data=df)
	p_hat_x = predict(binary_t_hat_func, df, type = "response")
	
	# Find the neighborhoods
	binary_neighbors = get.knn(X, k = k)
	
	# Get full results
	par_results = get_each_points_max_LRR_binary_parallel(X,binary_neighbors$nn.index, binary_y, binary_t, p_hat_x)
	
	max_LRR = sapply(par_results,function(x) x$max_LLR)
	normal_vectors=t(sapply(par_results,function(x) x$max_normal_vector))
	observed_prop_delta=sapply(par_results,function(x) x$observed_prop_delta)
	estimates = rbindlist(lapply(par_results,function(x) x$ix_estimates),fill=T)
	return(list(LRR=max_LRR,normal_vectors=normal_vectors,
							observed_prop_delta=observed_prop_delta,
							estimates=estimates))
}

run_LORD3_and_cbind_useful_output_parallel = function(X,Y,D,degree,k,nvec_col_names){
	# Run LORD3
	lord3_results = LORD3_binary_parallel(X,Y,D,degree,k)
	# Extract likelihood ratio, rename normal vector columns, bind into data.table
	LLR = lord3_results$LRR
	normal_vectors = lord3_results$normal_vectors
	colnames(normal_vectors) = nvec_col_names
	all_results = as.data.frame(cbind(LLR,normal_vectors))
	setDT(all_results)
	return(all_results)
}