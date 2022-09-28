library(data.table)
library(FNN)
library(LORD3)
library(tidyr)
library(dplyr)
library(Matrix)
library(stringr)
library(ggplot2)
library(RColorBrewer)

get_voroni_knn_discontinuities_index_sets_and_estimates = function(
		L_x, L_v, train_data, k_prime, t_partition, estimator,
		save_plots=NULL,R2_figure=F,vor_knn_ind = "vor_knn"
	){
	# Compute the Voroni KNN local RDs, and estimate CATE using `estimator`
	# at the projected centroid
	train_Xs = as.matrix(df[,colnames(L_x),with=F])

	U_w_index_set = forward_stepwise_voroni_neighborhoods(
		L_x,L_v,train_Xs,k_prime,t_partition,save_plots,R2_figure,vor_knn_ind
	)

	voroni_neighbors_and_sides = U_w_index_set$voroni_neighbors_and_sides
	U_x = U_w_index_set$U_x
	U_v = U_w_index_set$U_v

	# Get projected centroids.
	U_x = get_projected_centroids(U_x, U_v, train_Xs, voroni_neighbors_and_sides)

	# Use projected centroids as point locating CATE estimate.
	voroni_ball_TEs = get_voroni_ball_TE_estimates(U_x, U_v, voroni_neighbors_and_sides,
												   train_data,estimator)

	return(list(top_M_prime_TEs=voroni_ball_TEs,
			    neighbors_and_sides=voroni_neighbors_and_sides))

}

forward_stepwise_voroni_neighborhoods = function(L_x,L_v,train_Xs,k_prime,t_partition,
												 save_plots=NULL,R2_figure=F,vor_knn_ind = "vor_knn"){
	#- Select set of neighborhood centers U = {}
	#- For each neighborhood center x in L_x (in order):
	#    - Add x to U and recompute the Voronoi diagram.
	#    - For all centers x' in U, update the index set of x' to the intersection of
	#      k-neighborhood of x' and the Voronoi-neighborhood of x'.
	#    - If any x' does not have at least t_partition instances on each side of
	#    - its associated partitioning hyperplane, remove x (not x') from U  
	
	U_index = 1

	for (ix in 1:nrow(L_x)){

		if (ix == 1){
			U_x = L_x[ix,,drop=F]
			U_v = L_v[ix,,drop=F]
			U_prime_x = L_x[ix,,drop=F]
			U_prime_v = L_v[ix,,drop=F]
		} else {
			U_prime_x = rbind(U_x,L_x[ix,,drop=F])
			U_prime_v = rbind(U_v,L_v[ix,,drop=F])
		}		

		# Find voroni neighbors + sides, treating instances in U' as
		# true discontinuity points
		voroni_neighbors_and_sides = get_voroni_neighbors_knn_ball_intersection(
			U_prime_x,U_prime_v,train_Xs,k_prime,R2_figure, vor_knn_ind
		)
		# Check if they pass the threshold test
		passes_thresh_and_n_per_side = check_passes_thresholds(
			voroni_neighbors_and_sides,t_partition*2,t_partition
		)
		# If any don't pass, we exclude index ix from U. Otherwise we include.
		if (all(passes_thresh_and_n_per_side$passes_thresh)){
			U_x = U_prime_x
			U_v = U_prime_v
			if ((!is.null(save_plots))){
				p = plot_voroni_from_Xs_and_neighbors_and_sides(
					train_Xs,voroni_neighbors_and_sides
				)
				ggsave(paste0(save_plots,str_pad(U_index, 3, pad = "0"),'.png'))
				U_index = U_index + 1
			}
		}
	}
	# Finall rerun using U_prime if needed
	if (! all(passes_thresh_and_n_per_side$passes_thresh)){
		voroni_neighbors_and_sides = get_voroni_neighbors_knn_ball_intersection(
			U_x,U_v,train_Xs,k_prime,R2_figure, vor_knn_ind
		)
	}

	return(list(voroni_neighbors_and_sides=voroni_neighbors_and_sides,
				U_x = U_x, U_v = U_v))
}

get_voroni_neighbors_knn_ball_intersection = function(U_x,U_v,Xs,k,R2_figure,vor_knn_ind = "vor_knn"){
	# Computes the intersection of the KNN and Voroni neighborhoods,
	# and side assignments, for each point in U_x.

	neighbors_and_sides = list()

	# Get the Voroni neighborhood assignments
	neighborhood_assignment = get.knnx(U_x,Xs,k=1)$nn.index[,1]
	# Get the KNN ball assignments
	knn_assignments = get.knnx(Xs,U_x, k=k)$nn.index
	
	# Compute side assignments, using preferred orientation for visualization
	for (i in 1:nrow(U_x)){
		cx = U_x[i,]
		v = U_v[i,]

		# Find the intersection of the voroni and knn neighbors
		voroni_neighbors = which(neighborhood_assignment==i)
		knn_neighbors = knn_assignments[i,]
		if (vor_knn_ind == "vor_knn"){
			intersection_neighbors = intersect(voroni_neighbors,knn_neighbors)
			} else if (vor_knn_ind == "vor"){
			intersection_neighbors = voroni_neighbors
			}
			else {
			intersection_neighbors = knn_neighbors
		}
		
		# Partition across RD hyperplane
		cXs = center_neighborhood_ball(Xs,intersection_neighbors,cx)
		S = bisect_neighborhood(cXs,v)

		# We want to have coherent sides in figures visualizing Voroni KNN
		# index sets, so define preferred ordering of RD normal vector
		if (R2_figure){
			S = reorder_side(cXs,S)
		}
		
		neighbors_and_sides[[i]] = cbind(voroni_neighbors=intersection_neighbors,
										 side=S)
		
	}
	return(neighbors_and_sides)
}

reorder_side = function(cXs,S){
	# Redefine the side S to give visualizations visual coherency.
	side_df = as.data.frame(cbind(cXs,S))
	colnames(side_df) = c('X1','X2','S')
	grouped = side_df %>% 
				group_by(S) %>%
				summarise(m1=mean(X1), m2=mean(X2)) %>%
				ungroup()
	if (nrow(grouped)>1){
		grouped_delta_1 = grouped[grouped$S==1,]$m1 - grouped[grouped$S==0,]$m1
		grouped_delta_2 = grouped[grouped$S==1,]$m2 - grouped[grouped$S==0,]$m2
		if (abs(grouped_delta_2)>abs(grouped_delta_1)){
			if (grouped_delta_2 < 0){
				S = ifelse(S==0,1,0)
			}
		} else {
			if (grouped_delta_1 < 0){
				S = ifelse(S==0,1,0)
			}
		}
	}
	return(S)
}

check_passes_thresholds = function(voroni_neighbors_and_sides,total_thresh,min_side_thresh){
	# Check if voroni assignments all pass thresholds.
	n_possible_assignments = length(voroni_neighbors_and_sides)
	n_per_side = matrix(NA,nrow=n_possible_assignments,ncol=2)
	n_per_side[,1] = sapply(voroni_neighbors_and_sides,function(x) sum(x[,'side']==0))
	n_per_side[,2] = sapply(voroni_neighbors_and_sides,function(x) sum(x[,'side']==1))
	colnames(n_per_side) = c('0','1')
	n_per_side = as.data.frame(n_per_side)
	n_per_side$total = rowSums(n_per_side)
	passes_thresh = (n_per_side$total >= total_thresh) & apply(n_per_side[,c('0','1')] >= min_side_thresh,1,all)
	return(list(passes_thresh=passes_thresh,n_per_side=n_per_side))
}

plot_voroni_from_Xs_and_neighbors_and_sides = function(Xs,neighbors_and_sides,ncolors=50){
	# Plot the voroni + KNN neighborhoods
	ncolors = max(ncolors,length(neighbors_and_sides))
	getPalette = colorRampPalette(brewer.pal(9, "Set1"))(ncolors)
	set.seed(31534)
	permute_map = sample(ncolors,replace=F)
	with_Xs_df = list()
	for (i in 1:length(neighbors_and_sides)){
		x_ix = neighbors_and_sides[[i]][,'voroni_neighbors']
		with_Xs_df[[i]] = data.frame(Xs[x_ix,],side=neighbors_and_sides[[i]][,'side'],
									 nc=rep(i,length(x_ix)))
	}
	plot_data = rbindlist(with_Xs_df)
	plot_data$V = permute_map[plot_data$nc]
	p = ggplot(plot_data,aes(x=X1,y=X2,color=V,alpha=(side+0.5)/2)) + geom_point() +
		scale_color_gradientn(colors=getPalette,limits=c(0,ncolors)) +
		theme(legend.position = "none") + coord_equal() + xlim(c(0,1)) + ylim(c(0,1)) +
		xlab('X1') + ylab('X2') + ggtitle('Voroni union KNN index sets')
	return(p)
}

get_projected_centroids = function(U_x, U_v, train_Xs, voroni_neighbors_and_sides){
  # Uses the voroni neighbors to compute an updated neighborhood center -- e.g. the point at which
  # we are effectively estimating the heterogenous TE.
  
  new_centers = list()
  for (ix in 1:nrow(U_v)){
    
    n_center = unlist(U_x[ix,])
    cX = center_neighborhood_ball(train_Xs,
                                  voroni_neighbors_and_sides[[ix]][,'voroni_neighbors'],
                                  n_center)
    
    # Change of basis so X1 is aligned with the RD normal vector
	d = ncol(cX)
    A = matrix(rnorm(d^2),d,d)
    equals_col = apply(apply(A,2,function(x) x == U_v[ix,]),2,all)
    if (any(equals_col)){
      identical_col = which(equals_col)
      A = A[,c(identical_col,setdiff(1:ncol(A),identical_col))]
    }
    A[,1] = U_v[ix,]
    
    gs = gramSchmidt(A)
    Qt = t(gs$Q)
    rotated_cX = t(Qt %*% t(cX))
    
    # Now the first column of rotated_cX is along the nv dimension. To get the effective estimated
    # center, we average the remaining dimensions, then fix the first coordinate (along the normal vector)
    # to 0.
    
    voroni_centroid = colMeans(rotated_cX)
    voroni_centroid[1] = 0
    
    # Then we express using the original basis
    projected_mean_n_center = t(t(Qt) %*% as.matrix(voroni_centroid,nrow=2))
    
    # Note the projected mean n_center is the delta we need to use to shift the old center.
    new_centers[[ix]] = as.data.frame(n_center + projected_mean_n_center)
  }
  new_centers = rbindlist(new_centers)
  colnames(new_centers)  = colnames(train_Xs)
  return(new_centers)
}

get_voroni_ball_TE_estimates = function(U_x, U_v, voroni_neighbors_and_sides,
										train_data,estimator){
	# Get corresponding treatment effect estimates from the Voroni neighborhoods + side tables
	
	M_prime = nrow(U_x)
	CATEs = rep(NA,M_prime)
	ses = rep(NA,M_prime)
	dfs = rep(NA,M_prime)

	for (ix in 1:M_prime){
		s = voroni_neighbors_and_sides[[ix]][,'side']
		n_ix = voroni_neighbors_and_sides[[ix]][,'voroni_neighbors']
		ny = train_data[n_ix,Y]
		nX = as.matrix(train_data[n_ix,colnames(U_x),with=F])
		neigh_c = unlist(U_x[ix,])
		cX = center_neighborhood_ball(nX,1:nrow(nX),neigh_c)
		nt = train_data[n_ix,D]
		neigh_nv = unlist(U_v[ix,])
		est_f = get(estimator)

		estimate = est_f(s,ny,nt,cX,neigh_nv)

		CATEs[ix] = estimate$tau
		ses[ix] = estimate$se
		dfs[ix] = estimate$df
	}
	
	CATE_ests = cbind(U_x,U_v,
					  CATE = CATEs,
					  ses = ses,
					  lower = CATEs + qt(0.025,df=dfs)*ses,
					  upper = CATEs + qt(0.975,df=dfs)*ses,
					  df = dfs)
	
	return(CATE_ests)
}

