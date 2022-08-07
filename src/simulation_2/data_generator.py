import numpy as np
import pandas as pd

import torch
import gpytorch

from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from sklearn.exceptions import NotFittedError

from scipy.stats.distributions import norm
from scipy.stats import multivariate_normal

def get_bias_multiplier(z,gamma0,gamma1,gamma2):
    # Computes the bias multiplier depending on the region
    z_map = {0:gamma0,
             1:gamma1,
             2:gamma2}

    effective_thresh = z_map[z]
    return -(norm.pdf(effective_thresh)/norm.cdf(effective_thresh) + 
                norm.pdf(effective_thresh)/(1-norm.cdf(effective_thresh)))

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class InducedPosteriorFunc(object):
    def __init__(self,hypers,n_inducing_points=50):

        # Need to init with X, Y but will just sample in prior mode
        inducing_X0, inducing_X1 = torch.meshgrid(torch.linspace(0,1,n_inducing_points),torch.linspace(0,1,n_inducing_points))
        self.inducing_X = torch.cat([inducing_X0.flatten().unsqueeze(-1),inducing_X1.flatten().unsqueeze(-1)],1)
        _inducing_Y = torch.tensor(np.random.randn(self.inducing_X.shape[0]), dtype=torch.float)
        
        self.hypers = hypers
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = ExactGPModel(self.inducing_X,_inducing_Y,self.likelihood)
        self.gp.initialize(**self.hypers)

        self.gp.eval()
        self.likelihood.eval()

        with gpytorch.settings.prior_mode(True):
            prior_sample = self.gp(self.inducing_X)

        self.inducing_Y = prior_sample.rsample()
        
        self.gp.set_train_data(inputs=self.inducing_X,targets=self.inducing_Y)

    def f(self,X):
        
        with torch.no_grad() and gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=False):
            Y = self.gp(X).mean.detach().numpy()
            
        return Y

class DataGenerator(object):
    def __init__(self,N,p,gamma0,gamma1,gamma2,max_abs_rho,CATE_hypers,rhosigma_hypers):
        self.N = N
        self.p = p
        
        self.max_abs_rho = max_abs_rho
        
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        self.CATE_hypers = CATE_hypers
        self.rhosigma_hypers = rhosigma_hypers
        
    def draw_CATE_and_rhosigma_f(self):
        self.CATE_fun = InducedPosteriorFunc(self.CATE_hypers)
        self.rhosigma_fun = InducedPosteriorFunc(self.rhosigma_hypers)
        
    def set_sigma(self,train_rhosigma):
        self.sigma = np.abs(train_rhosigma).max() / self.max_abs_rho

    def get_bias_multiplier(self,z):
        # Computes the bias multiplier depending on the region
        z_map = {0:self.gamma0,
                 1:self.gamma1,
                 2:self.gamma2}

        effective_thresh = z_map[z]
        return -(norm.pdf(effective_thresh)/norm.cdf(effective_thresh) + 
                 norm.pdf(effective_thresh)/(1-norm.cdf(effective_thresh)))

    def get_Z(self,X):
        # Two RD instruments, with shared latent index variable
        Z1 = (X.max(1).values>0.5) & (X.min(1).values<0.5)
        Z2 = (X.min(1).values>0.5)
        Z = Z1 + 2*Z2
        return Z.numpy()

    def get_U(self,rho,sigma):
        U = np.zeros((len(rho),2))

        for ix in range(U.shape[0]):

            cov = np.array([[1,             rho[ix]*sigma],
                            [rho[ix]*sigma, sigma**2]])

            U[ix,] = multivariate_normal.rvs(cov=cov)

        U = np.concatenate([U,U[:,[1]]],1)
        return U
    
    def get_D(self,U,Z):
        Ud = U[:,0]
        
        # Three latent treatment variables
        D0 = (self.gamma0>Ud)
        D1 = (self.gamma1>Ud)
        D2 = (self.gamma2>Ud)

        # Observed treatment
        D = D2*(Z==2) + D1*(Z==1) + D0*(Z==0)
        
        return D
    
    def get_potential_outcomes(self,tau,U,D):
        Y1 = 1./2.*tau + U[:,2]
        Y0 = -1./2.*tau + U[:,1]
        Y = Y1*D + Y0*(1-D)
        
        return Y0,Y1,Y
        
    
    def sample_train_data(self):
        
        if not hasattr(self,'CATE_fun'):
            raise NotFittedError('Call draw_CATE_and_rhosigma before sampling, to fix functions.')
        
        # Uniformly sample X
        X = torch.tensor(np.random.rand(self.N,self.p), dtype=torch.float)
        tau = self.CATE_fun.f(X)
        rhosigma = self.rhosigma_fun.f(X)
        
        # Fit sigma to bound training rho in [-self.max_abs_rho,self.max_abs_rho]
        self.set_sigma(rhosigma)
        rho = rhosigma/self.sigma
        
        # Sample U 
        U = self.get_U(rho,self.sigma)
        
        # Two RD instruments, with shared latent index variable
        Z = self.get_Z(X)
        
        # Compute potential treatment indicators 
        D = self.get_D(U,Z)
        
        # Computed expected bias
        Z_bias_multiplier = np.array([self.get_bias_multiplier(z.item()) for z in Z])
        bias = self.sigma*rho*Z_bias_multiplier
    
        # Compute potential outcomes
        Y0,Y1,Y = self.get_potential_outcomes(tau,U,D)
        
        return X,U,D,Z,Y0,Y1,Y,bias,tau
        
        
    def compute_CATE_and_bias_at_X(self,X):
        
        
        if not hasattr(self,'sigma'):
            raise NotFittedError('Need to sample training data to fix sigma.')
        
        
        tau = self.CATE_fun.f(X)
        Z = self.get_Z(X)
        Z_bias_multiplier = np.array([self.get_bias_multiplier(z.item()) for z in Z])
        rhosigma = self.rhosigma_fun.f(X)
        rho = rhosigma/self.sigma
        bias = self.sigma*rho*Z_bias_multiplier
        
        return tau, bias