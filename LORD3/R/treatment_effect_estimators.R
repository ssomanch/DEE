library(AER)
library(data.table)
library(pracma)


nonparametric_estimator = function(group, y, t,...){
  # The non-parametric estimator is just a 2sls estimator with no covariates.
  # D is treatment indicator; S is discovered RD instrument.
  two_SLS_formula = 'Y ~ D | S'

  if (length(unique(group))==2){
    reg_df = data.frame(S=group,Y=y,D=t)
    est = ivreg(two_SLS_formula,data=reg_df)
    tau = est$coefficients['D']

    # If weak instrument, then tau can be NA. Note this isn't covered
    # by the check that LLR is finite, since LLR based on delta over
    # background model.
    if (! is.na(tau)){
      se = coeftest(est, vcov. = vcov(est), df = est$df.residual)['D','Std. Error']
      df = df.residual(est)
    } else {
      se = NA
      df = NA
    }
  } else {
    se = NA
    tau = NA
    df = NA
  }

  return(list(tau=tau,se=se,df=df))
}


noninteracted_2SLS = function(group, y, t, cX,...){
  # The fuzzy RD estimator is a 2sls with linear control specifications.
  # Here cX are the centered x variables
  two_SLS_formula = 'Y ~ D + X1 + X2 | X1 + X2 + S'


  if (length(unique(group))==2){
    reg_df = data.frame(S=group,Y=y,D=t, X1=cX[,1], X2=cX[,2])
    est = ivreg(two_SLS_formula,data=reg_df)
    tau = est$coefficients['D']

    # If weak instrument, then tau can be NA. Note this isn't covered
    # by the check that LLR is finite, since LLR based on delta over
    # background model.
    if (! is.na(tau)){
      se = coeftest(est, vcov. = vcov(est), df = est$df.residual)['D','Std. Error']
      df = df.residual(est)
    } else {
      se = NA
      df = NA
    }
  } else {
    se = NA
    tau = NA
    df = NA
  }

  return(list(tau=tau,se=se,df=df))
}


interacted_2SLS = function(group, y, t, cX,...){
  # The fuzzy RD estimator is a 2sls but now we're letting the
  # coefficients vary across the learned partition.
  # See equations 4.8/4.9 in https://www.nber.org/papers/w13039.pdf
  two_SLS_formula = 'Y ~ D + X1L + X1R + X2L + X2R | X1L + X1R + X2L + X2R + S'


  if (length(unique(group))==2){
    reg_df = data.frame(S=group,Y=y,D=t,
    										X1L=cX[,1]*group, X1R=cX[,1]*(1-group),
    										X2L=cX[,2]*group, X2R=cX[,2]*(1-group))
    est = ivreg(two_SLS_formula,data=reg_df)
    tau = est$coefficients['D']

    # If weak instrument, then tau can be NA. Note this isn't covered
    # by the check that LLR is finite, since LLR based on delta over
    # background model.
    if (! is.na(tau)){
      se = coeftest(est, vcov. = vcov(est), df = est$df.residual)['D','Std. Error']
      df = df.residual(est)
    } else {
      se = NA
      df = NA
    }
  } else {
    se = NA
    tau = NA
    df = NA
  }

  return(list(tau=tau,se=se,df=df))
}

rotated_2SLS = function(group, y, t, cX, nv){
  # The fuzzy RD estimator is a 2sls, but where we rotate the centered matrix
	# so the RD normal vector is axis aligned
  two_SLS_formula = 'Y ~ D + cXnvL + cXnvR | cXnvL + cXnvR + S'

  # Project onto normal vector
  scalar_proj = cX %*% (nv / sqrt(sum(nv^2)))


  if (length(unique(group))==2){
    reg_df = data.frame(S=group,Y=y,D=t,
    										cXnvL=scalar_proj*group, cXnvR=scalar_proj*(1-group))
    est = ivreg(two_SLS_formula,data=reg_df)
    tau = est$coefficients['D']

    # If weak instrument, then tau can be NA. Note this isn't covered
    # by the check that LLR is finite, since LLR based on delta over
    # background model.
    if (! is.na(tau)){
      se = coeftest(est, vcov. = vcov(est), df = est$df.residual)['D','Std. Error']
      df = df.residual(est)
    } else {
      se = NA
      df = NA
    }
  } else {
    se = NA
    tau = NA
    df = NA
  }

  return(list(tau=tau,se=se,df=df))
}
