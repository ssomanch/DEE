###############################################################
# Utilities for the simplest case, where the CEFs for Y0/Y1=0 #

get_complier_shift = function(sigma, gamma0, gamma1){
  # Shift from the CATE to the LATE given the normal noise
  # covariance matrix
  rho0 = sigma[1,2]/sqrt(sigma[1,1]*sigma[2,2])
  rho1 = sigma[1,3]/sqrt(sigma[1,1]*sigma[3,3])
  sigma0 = sqrt(sigma[2,2])
  sigma1 = sqrt(sigma[3,3])

  return((rho1*sigma1 - rho0*sigma0)*(dnorm(gamma0)-dnorm(gamma1))/(pnorm(gamma1)-pnorm(gamma0)))
}

expected_U0 = function(sigma, gamma0, gamma1, includes_compliers){
  rho0 = sigma[1,2]/sqrt(sigma[1,1]*sigma[2,2])
  sigma0 = sqrt(sigma[2,2])
  if (includes_compliers){
    return(rho0*sigma0*(dnorm(gamma0)/(1 - pnorm(gamma0))))
  } else {
    return(rho0*sigma0*(dnorm(gamma1)/(1 - pnorm(gamma1))))
  }
}

expected_U1 = function(sigma, gamma0, gamma1, includes_compliers){
  rho1 = sigma[1,3]/sqrt(sigma[1,1]*sigma[3,3])
  sigma1 = sqrt(sigma[3,3])
  if (includes_compliers){
    return(-rho1*sigma1*(dnorm(gamma1)/ pnorm(gamma1)))
  } else {
    return(-rho1*sigma1*(dnorm(gamma0)/ pnorm(gamma0)))
  }
}

observational_estimator_expected_bias = function(x,sigma,gamma0,gamma1,b){
  if(all(x>b)){
    # Then compliers are treated
    return(expected_U1(sigma, gamma0, gamma1, includes_compliers = T) - expected_U0(sigma, gamma0, gamma1, includes_compliers = F))
  } else {
    # Then compliers are not treated
    return(expected_U1(sigma, gamma0, gamma1, includes_compliers = F) - expected_U0(sigma, gamma0, gamma1, includes_compliers = T))
  }
}

sample_sigma = function(s){
  # Given a seed s, generate a valid covariance matrix
  set.seed(s)
  M = matrix(runif(9)*10-5,nrow=3)
  sigma = t(M) %*% M
  scale_factor = sqrt(sigma[1,1])
  M = M*1/scale_factor
  sigma = t(M) %*% M
  return(sigma)
}

test_seed = function(s,gamma0,gamma1){
  # Generates a covariance matrix and returns
  # a (scaled) measure of the deviation between
  # the CATE and LATE
  sigma = sample_sigma(s)
  shift = get_complier_shift(sigma,gamma0,gamma1)
  scale = norm(sigma,'f') # Scale sigma by Frobenius norm
  return(shift/scale)
}


make_conditional_constant_effect_sigma = function(rho0, sigma0){
  matrix(c(1,sigma0*rho0,sigma0*rho0,
           sigma0*rho0, sigma0**2, sigma0**2,
           sigma0*rho0, sigma0**2, sigma0**2),
         nrow=3,ncol=3)
}

##########################################################
# Helper functions to sample unobservables so bias in
# observational estimator is non-constant. These are
# legacy -- they're no super obvious functions.
##########################################################

sample_UD01_given_X = function(X,gamma0,gamma1,b){
  rho_and_sigma = matrix(c(compute_rho_given_X(X),compute_sigma_given_X(X)),ncol=2)
  U01d = matrix(nrow = nrow(X),ncol=3)
  mu = c(0,0,0)
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    U01d[i,] = rmvnorm(1,mean=mu,sigma=sigma)
  }
  return(U01d)
}

compute_rho_given_X = function(X){
  return(exp(X[,1]-1)*0.9)
}

compute_sigma_given_X = function(X){
  return((X[,2]+1)**2)
}

compute_oe_bias_given_X = function(X,gamma0,gamma1,b){
  rho_and_sigma = matrix(c(compute_rho_given_X(X),compute_sigma_given_X(X)),ncol=2)
  expected_bias = rep(NA,nrow(X))
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    expected_bias[i] = observational_estimator_expected_bias(
      X[i,],sigma,gamma0,gamma1,b
    )
  }
  return(expected_bias)
}

##########################################################
# Helper functions to conditionally sample unobservables
# so bias in the observational estimator is non-constant.
# These are second gen -- they yield nice linear bias fun.
##########################################################

compute_rho_given_X_linear = function(X,sigmacoef){
  return(sqrt(rowSums(X)/2))
}

compute_sigma_given_X_linear = function(X,sigmacoef){
  return(sigmacoef*sqrt(rowSums(X)))
}

compute_oe_bias_given_X_linear = function(X,gamma0,gamma1,b,sigmacoef){
  rho_and_sigma = matrix(c(compute_rho_given_X_linear(X),
                           compute_sigma_given_X_linear(X,sigmacoef)),
                         ncol=2)
  expected_bias = rep(NA,nrow(X))
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    expected_bias[i] = observational_estimator_expected_bias(
      X[i,],sigma,gamma0,gamma1,b
    )
  }
  return(expected_bias)
}

compute_expected_Y = function(x,sigma,gamma0,gamma1,b,ycoef){
  if(all(x>b)){
    # Then compliers are treated
    return(pnorm(gamma1)*(sum(x)*ycoef+expected_U1(sigma, gamma0, gamma1, includes_compliers = T)) +
             (1-pnorm(gamma1))*expected_U0(sigma, gamma0, gamma1, includes_compliers = F))
  } else {
    # Then compliers are not treated
    return(pnorm(gamma0)*(sum(x)*ycoef+expected_U1(sigma, gamma0, gamma1, includes_compliers = F)) +
             (1-pnorm(gamma0))*expected_U0(sigma, gamma0, gamma1, includes_compliers = T))
  }
}

compute_EY_given_X_linear = function(X,gamma0,gamma1,b,sigmacoef,ycoef){
  rho_and_sigma = matrix(c(compute_rho_given_X_linear(X),
                           compute_sigma_given_X_linear(X,sigmacoef)),
                         ncol=2)
  EY = rep(NA,nrow(X))
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    EY[i] = compute_expected_Y(
      X[i,],sigma,gamma0,gamma1,b,ycoef
    )
  }
  return(EY)
}

compute_sigma_01d_given_X_linear = function(X,gamma0,gamma1,b,sigmacoef){
  rho_and_sigma = matrix(c(compute_rho_given_X_linear(X),
                           compute_sigma_given_X_linear(X,sigmacoef)),
                         ncol=2)
  sigma_01d = rep(NA,nrow(X))
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    sigma_01d[i] = sigma[1,2]
  }
  return(sigma_01d)
}

sample_UD01_given_X_linear = function(X,sigmacoef){
  rho_and_sigma = matrix(c(compute_rho_given_X_linear(X),compute_sigma_given_X_linear(X,sigmacoef)),ncol=2)
  U01d = matrix(nrow = nrow(X),ncol=3)
  mu = c(0,0)
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    u_d_01 = rmvnorm(1,mean=mu,sigma=sigma[1:2,1:2])
    U01d[i,1] = u_d_01[1]
    U01d[i,2] = u_d_01[2]
    U01d[i,3] = u_d_01[2]
  }
  return(U01d)
}


##########################################################
# Third gen -- this yields a linear bias function with
# lower noise along the discontinuity + stronger signal
# (e.g. a steeper linear slope in the bias)
##########################################################

compute_rho_given_X_linear_new = function(X){
  return(sign(rowSums(X)-1)*sqrt(abs(rowSums(X)-1)))
}

compute_sigma_given_X_linear_new = function(X,sigmacoef,eps=1e-8){
  return(sigmacoef*sqrt(abs(rowSums(X)-1)) + eps)
}

compute_oe_bias_given_X_linear_new = function(X,gamma0,gamma1,b,sigmacoef){
  rho_and_sigma = matrix(c(compute_rho_given_X_linear_new(X),
                           compute_sigma_given_X_linear_new(X,sigmacoef)),
                         ncol=2)
  expected_bias = rep(NA,nrow(X))
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    expected_bias[i] = observational_estimator_expected_bias(
      X[i,],sigma,gamma0,gamma1,b
    )
  }
  return(expected_bias)
}

sample_UD01_given_X_linear_new = function(X,sigmacoef){
  rho_and_sigma = matrix(c(compute_rho_given_X_linear_new(X),compute_sigma_given_X_linear_new(X,sigmacoef)),ncol=2)
  U01d = matrix(nrow = nrow(X),ncol=3)
  mu = c(0,0)
  for (i in 1:nrow(X)){
    sigma = make_conditional_constant_effect_sigma(rho_and_sigma[i,1],
                                                   rho_and_sigma[i,2])
    u_d_01 = rmvnorm(1,mean=mu,sigma=sigma[1:2,1:2])
    U01d[i,1] = u_d_01[1]
    U01d[i,2] = u_d_01[2]
    U01d[i,3] = u_d_01[2]
  }
  return(U01d)
}
