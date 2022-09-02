
##########################################################
# Balance check functions
##########################################################

run_balance_one_outcome_np = function(bound,cX,S,neighbors,nv,balance_col){
  # Run reduced form
  reduced_form_eq = 'Y ~ S'

  y = bound[[balance_col]][neighbors]
  
  reg_df = data.frame(S=as.integer(S),Y=y)
  est = lm(reduced_form_eq,data=reg_df)

  summ = summary(est)
  p = summ$coefficients['S',4]
  return(p)
}

run_balance_one_outcome = function(bound,cX,S,neighbors,nv,balance_col){
  # Run 2SLS
  reduced_form_eq = 'Y ~ S + cXnvL + cXnvR'
  y = bound[[balance_col]][neighbors]

  # Project onto normal vector
  scalar_proj = cX %*% (nv / sqrt(sum(nv^2)))

  reg_df = data.frame(S=as.integer(S),Y=y,cXnvL=scalar_proj*S, cXnvR=scalar_proj*(1-S))
  est = lm(reduced_form_eq,data=reg_df)

  summ = summary(est)
  p = summ$coefficients['S',4]
  return(p)
}

run_balance_one_neighborhood = function(Xs,knns,bound,ix,control_cols,balance_est){

  cx = Xs[ix,]
  nv = t(as.matrix(bound[ix,.(longitude_nvec,latitude_nvec,total_pop_nvec)]))
  neighbors = knns$nn.index[ix,]

  # Partition across RD hyperplane
  cX = as.matrix(center_neighborhood_ball(Xs,neighbors,cx))
  S = bisect_neighborhood(cX,nv)
  
  out = sapply(control_cols,function(x) balance_est(bound,cX,S,neighbors,nv,x))
  return(out)
}
               

run_balance_one_VKNN_neighborhood = function(Xs,voroni_assets,df,ix,control_cols,balance_est){

  cx = as.matrix(voroni_assets$top_M_prime_TEs)[ix,c('longitude','latitude','total_pop')]
  nv = as.matrix(voroni_assets$top_M_prime_TEs)[ix,c('longitude_nvec','latitude_nvec','total_pop_nvec')]
  neighbors = voroni_assets$neighbors_and_sides[[ix]][,'voroni_neighbors']

  # Partition across RD hyperplane
  cX = as.matrix(center_neighborhood_ball(Xs,neighbors,cx))
  S = voroni_assets$neighbors_and_sides[[ix]][,'side']
  
  out = sapply(control_cols,function(x) balance_est(df,cX,S,neighbors,nv,x))
  return(out)
}