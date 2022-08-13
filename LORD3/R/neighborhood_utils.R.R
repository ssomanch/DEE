# Functions for computing KNN neighborhoods

center_neighborhood_ball = function(X, neighbors_indices, centroid){
  # Returns a centered KNN (e.g. subtracting the neighborhood
  # center from each point in the ball).
  if (length(neighbors_indices)>1){
    return(sweep(X[neighbors_indices,],2,centroid,'-'))
  } else {
    return(X[neighbors_indices,] - centroid)
  }
}

bisect_neighborhood = function(cneigh, normal_vector){
  # This is inefficient -- can just take n*%*t(n) (e.g. dxk *%* kxd)
  # and then get all of the groups in a matrix.
  return((cneigh %*% normal_vector >= 0)[,1])
}

all_groups = function(cneigh){
  # Efficient version is just matrix multiplication AA^T
  return((cneigh %*% t(cneigh)) >= 0)
}

