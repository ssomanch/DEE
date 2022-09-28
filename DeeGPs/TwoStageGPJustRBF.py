import numpy as np
import pandas as pd

from torch import Tensor
import gpytorch
import botorch
import torch

from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from gpytorch.metrics import negative_log_predictive_density, mean_standardized_log_loss, quantile_coverage_error
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior 
from gpytorch.constraints.constraints import GreaterThan

from sklearn.metrics import pairwise_distances

from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

MIN_SE = 1e-2

class ExactGPModelJustRBF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelJustRBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TwoStageGPJustRBF(object):
    r"""A single-task exact GP model using a heteroskeastic noise model, with
    a Gaussian kernel, where we fit in two stages:
        - First fitting the SE noise model
        - Then fitting the TE model
    The posterior predictive thus uses the fixed noise SingleTaskGP, where we
    push through the mean noise estimate from the stage 1 model.
    """

    def __init__(self,train_x,train_y,observed_var):
        
        self.noise_likelihood = GaussianLikelihood()
        self.noise_model = ExactGPModelJustRBF(
            train_x=train_x,
            train_y=observed_var,
            likelihood=self.noise_likelihood
        )
        
        self.TE_likelihood = FixedNoiseGaussianLikelihood(
            noise=observed_var,
            learn_additional_noise=False
        )
        #self.TE_likelihood = GaussianLikelihood()

        self.TE_model = ExactGPModelJustRBF(
            train_x=train_x, train_y=train_y.flatten(), likelihood=self.TE_likelihood
        )

        self.noise_model.train()
        self.noise_likelihood.train()
        self.TE_model.train()
        self.TE_likelihood.train()

    def fit(self,train_x,train_y,observed_var,loo=False):
        
        if not loo:
            self.noise_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.noise_model.likelihood,
                                                                      self.noise_model)
        else:
            self.noise_mll = LeaveOneOutPseudoLikelihood(self.noise_model.likelihood,
                                                         self.noise_model)
        self.noise_mll.train()
        fit_noise_model = botorch.fit.fit_gpytorch_model(self.noise_mll)
        
        if not loo:
            self.TE_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.TE_model.likelihood, self.TE_model)
        else:
            self.TE_mll = LeaveOneOutPseudoLikelihood(self.TE_model.likelihood, self.TE_model)
        
        self.TE_mll.train()
        fit_TE_model = botorch.fit.fit_gpytorch_model(self.TE_mll)
        
        self.noise_mll.eval()
        self.noise_model.eval()
        self.noise_likelihood.eval()
        self.TE_mll.eval()
        self.TE_model.eval()
        self.TE_likelihood.eval()

class TwoStageGPJustRBFWrapper(object):
    def fit(self,train_x,train_y,observed_var,loo=False):
        
        self.model = TwoStageGPJustRBF(train_x,
                                       train_y,
                                       observed_var)
        
        # Stage one: Fit the noise model
        self.model.fit(train_x,train_y,observed_var,loo)

    def noise_posterior(self,test_x):
        with torch.no_grad():
            posterior = self.model.noise_model(test_x)
            lower_p, upper_p = posterior.confidence_region()
            mu = posterior.mean
        to_plot = np.concatenate([test_x.numpy(),
                                  lower_p.numpy().reshape(-1,1),
                                  upper_p.numpy().reshape(-1,1),
                                  mu.numpy().reshape(-1,1)],axis=1)
        to_plot = pd.DataFrame(to_plot)
        to_plot.columns = [f"X{i}" for i in range(1,test_x.shape[1]+1)] + ['lower','upper','mu']

        return to_plot
    
    def posterior(self,test_x):
        with torch.no_grad():
            posterior = self.model.TE_model(test_x)
            lower_p, upper_p = posterior.confidence_region()
            mu = posterior.mean
        to_plot = np.concatenate([test_x.numpy(),
                                  lower_p.numpy().reshape(-1,1),
                                  upper_p.numpy().reshape(-1,1),
                                  mu.detach().numpy().reshape(-1,1)],axis=1)
        to_plot = pd.DataFrame(to_plot)
        to_plot.columns = [f"X{i}" for i in range(1,test_x.shape[1]+1)] + ['lower','upper','mu']

        return to_plot

    def posterior_predictive(self,test_x):
        noise_posterior_mean = torch.clamp(torch.tensor(self.noise_posterior(test_x).mu.values),
                                           MIN_SE)
        with torch.no_grad():
            predictive_posterior = self.model.TE_likelihood(
                self.model.TE_model(test_x), noise=noise_posterior_mean.flatten()
            )
            # predictive_posterior = self.model.TE_likelihood(
            #     self.model.TE_model(test_x), noise=(torch.tensor(np.repeat(0.01, len(noise_posterior_mean)), dtype=torch.float)**2).flatten()
            # )
            
            # predictive_posterior = self.model.TE_likelihood(
            #     self.model.TE_model(test_x))
            lower_predictive, upper_predictive = predictive_posterior.confidence_region()
            mu = predictive_posterior.mean
            
        to_plot = np.concatenate([test_x.numpy(),
                                  lower_predictive.numpy().reshape(-1,1),
                                  upper_predictive.numpy().reshape(-1,1), 
                                  mu.detach().numpy().reshape(-1,1)],axis=1)
        to_plot = pd.DataFrame(to_plot)
        to_plot.columns = [f"X{i}" for i in range(1,test_x.shape[1]+1)] + ['lower','upper','mu']

        return to_plot

    def describe(self):
        with torch.no_grad():
            description = {
                'Noise model RBF lengthscale': self.model.noise_model.covar_module.base_kernel.lengthscale[0].item(),
                'Noise model RBF outputscale': self.model.noise_model.covar_module.outputscale.item(),
                'Noise model mean': self.model.noise_model.mean_module.constant.item(),
                'f* model mean': self.model.TE_model.mean_module.constant.item(),
                'f* model RBF lengthscale': self.model.TE_model.covar_module.base_kernel.lengthscale[0].item(),
                'f* model RBF outputscale': self.model.TE_model.covar_module.outputscale.item()
            }
        return description

    def get_lml(self,train_x,train_y,observed_var):
        self.model.TE_model.train()
        self.model.TE_likelihood.train()

        lml = self.model.TE_likelihood(self.model.TE_model(train_x),noise=observed_var).log_prob(train_y)

        self.model.TE_model.eval()
        self.model.TE_likelihood.eval()

        return lml.item()
    
    def get_causal_mse(self,test_x,test_cf_phats,true_te):
        with torch.no_grad():
            posterior = self.model.TE_model(test_x)
            mu = posterior.mean
        
        bias_est = mu.numpy().flatten()
        debiased_est = test_cf_phats - bias_est
        causal_mse = np.mean((debiased_est - true_te)**2)
        return causal_mse
    
    def get_posterior_predictive_marginal_mse(self,test_x,test_cf_phats,true_te):
        noise_posterior_mean = torch.clamp(torch.tensor(self.noise_posterior(test_x).mu.values),
                                           MIN_SE)
        with torch.no_grad():
            predictive_posterior = self.model.TE_likelihood(
                self.model.TE_model(test_x), noise=noise_posterior_mean.flatten()
            )
            post_var = predictive_posterior.stddev.numpy().flatten()**2
            post_mu = predictive_posterior.mean.numpy()
        
        o_b = test_cf_phats - true_te
        posterior_marginal_mse = post_var + post_mu*(post_mu-2*o_b) + o_b**2
        return posterior_marginal_mse.mean()

    def get_posterior_predictive_metrics(self, test_x, true_te, quantile = 95):
        noise_posterior_mean = torch.clamp(torch.tensor(self.noise_posterior(test_x).mu.values),
                                           MIN_SE)
        with torch.no_grad():
            predictive_posterior = self.model.TE_likelihood(
                self.model.TE_model(test_x), noise=noise_posterior_mean.flatten()
            )
            true_te_tensor = torch.from_numpy(true_te).float()
            nlpd = negative_log_predictive_density(predictive_posterior, true_te_tensor)
            msll = mean_standardized_log_loss(predictive_posterior, true_te_tensor)
            qce = quantile_coverage_error(predictive_posterior, true_te_tensor, quantile=quantile)
            
            print(predictive_posterior.log_prob(true_te_tensor)/true_te_tensor[-1], true_te_tensor.shape[-1])
        
        return {'nlpd': nlpd.item(), 
                'msll': msll.item(), 
                'qce': qce.item()}
        
    def get_CV_PLP(self,train_x,train_y,observed_var,k=None,thresh=None):
        
        dists = pairwise_distances(train_x)
        
        PLP = 0

        for ix in range(train_x.shape[0]):
            if thresh is None:
                holdout_ix = np.argsort(dists[ix,:])[:k]
                train_ix = np.argsort(dists[ix,:])[k:]
            else:
                ix_thresh = min(thresh,dists[ix,:].max())
                holdout_ix = dists[ix,:] < ix_thresh
                train_ix = ~holdout_ix

            _train_x = train_x[train_ix,:]
            _train_y = train_y[train_ix]
            _train_noise = observed_var[train_ix]

            x_ix = train_x[[ix],:]
            y_ix = train_y[ix]
            noise_ix = observed_var[ix]

            self.model.TE_model.set_train_data(inputs=_train_x,targets=_train_y,strict=False)
            self.model.TE_likelihood.noise = _train_noise
            PLP += self.model.TE_likelihood(self.model.TE_model(x_ix),noise=noise_ix).log_prob(y_ix)

        self.model.TE_model.set_train_data(inputs=train_x,targets=train_y,strict=False)
        self.model.TE_likelihood.noise = observed_var
        return PLP.item()