import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.metrics import pairwise_distances

#########################################
# Utility functions for PLP BMA         #
#########################################

def random_dist_PLP_and_candidate_weights(gp,train_x,train_y,train_noise,ix,d_dist):
    """Compute BLOOCV posterio log prob (PLP) for training instance ix, holding out its k-NN.
    """
    d = np.random.choice(d_dist,1)
    dists = pairwise_distances(train_x)
    train_ix = (dists[ix,:] >= d).nonzero()[0]
    _train_x = train_x[train_ix,:]
    _train_y = train_y[train_ix]
    _train_noise = train_noise[train_ix]

    x_ix = train_x[[ix],:]
    y_ix = train_y[ix]
    noise_ix = train_noise[ix]

    gp.model.TE_model.set_train_data(inputs=_train_x,targets=_train_y,strict=False)
    gp.model.TE_likelihood.noise = _train_noise
    
    if (_train_x.shape[0]>0):
        gp.model.TE_model.eval()
        gp.model.TE_likelihood.eval()
        posterior_pred = gp.model.TE_likelihood(gp.model.TE_model(x_ix),noise=noise_ix).log_prob(y_ix)
    else:
        with gpytorch.settings.prior_mode(True):
            posterior_pred = gp.model.TE_likelihood(gp.model.TE_model(x_ix),noise=noise_ix).log_prob(y_ix)

    gp.model.TE_model.eval()
    gp.model.TE_likelihood.eval()
    gp.model.TE_model.set_train_data(inputs=train_x,targets=train_y,strict=False)
    gp.model.TE_likelihood.noise = train_noise
    
    return posterior_pred.item()

def k_holdout_PLP_and_candidate_weights(gp,train_x,train_y,train_noise,ix,k):
    """Compute the two candidate weights -- minimum distance to a training
    instance, and posterior standard deviation -- for GP holding out
    the k nearest instances.
    """
    dists = pairwise_distances(train_x)
    train_ix = np.argsort(dists[ix,:])[k:]
    
    _train_x = train_x[train_ix,:]
    _train_y = train_y[train_ix]
    _train_noise = train_noise[train_ix]

    x_ix = train_x[[ix],:]
    y_ix = train_y[ix]
    noise_ix = train_noise[ix]

    gp.model.TE_model.set_train_data(inputs=_train_x,targets=_train_y,strict=False)
    gp.model.TE_likelihood.noise = _train_noise
    stddev = gp.model.TE_model(x_ix).stddev.item()
    posterior_pred = gp.model.TE_likelihood(gp.model.TE_model(x_ix),noise=noise_ix).log_prob(y_ix)
    
    min_d = (dists[ix,:][train_ix]).min()
    
    return {'stddev':stddev,'PLP':posterior_pred.item(),'min dist':min_d}

def get_weighted_plp(gp,train_x,train_y,train_noise,ix,test_weight_target,weight_col):
    """Compute the weighted PLP, weighting to approximately match the test distribution
    of the metric identified by weight_col (either stddev or min dist).
    """
    weights_and_PLP = pd.DataFrame([
        k_holdout_PLP_and_candidate_weights(gp,train_x,train_y,train_noise,ix,k) \
        for k in range(train_x.shape[0])
    ])
    
    is_duplicated = weights_and_PLP[weight_col].duplicated()
    weights_and_PLP = weights_and_PLP.loc[~is_duplicated]
    bins = pd.cut(test_weight_target,[-1] + weights_and_PLP[weight_col].tolist())
    weights = pd.value_counts(bins,normalize=True).sort_index()
    weights_and_PLP['weight'] = weights.values
    weights_and_PLP['weighted_plp'] = weights_and_PLP.weight * weights_and_PLP.PLP
    return weights_and_PLP['weighted_plp'].sum()

def get_loo_plp(gp,train_x,train_y,train_noise,ix):
    """Vanilla LOO PLP.
    """
    stddevs_and_plp = k_holdout_PLP_and_candidate_weights(gp,train_x,train_y,train_noise,ix,1)
    return stddevs_and_plp['PLP']