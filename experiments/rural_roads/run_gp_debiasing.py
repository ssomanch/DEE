import pandas as pd
import numpy as np
import scipy
import sys
import os
from sklearn.metrics import pairwise_distances

import sys
sys.path.append(os.getcwd())
from DeeGPs.PLP_BMA_utils import *

from torch import Tensor
import gpytorch
import botorch
import torch
from botorch.optim.fit import fit_gpytorch_torch

from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

# Plotting libraries

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('paper',font_scale=1.4)

import argparse

parser = argparse.ArgumentParser(description='GP debiasing estimator for roads dataset')
parser.add_argument('--y', help='Outcome variable')
parser.add_argument('--RD-type', help='Which type of RDs to include')
parser.add_argument('--estimator', help='VKNN estimator')
parser.add_argument('--k-prime', help='k prime')
parser.add_argument('--t-partition', help='t partition')

args = parser.parse_args()
y_name = args.y
RD_type = args.RD_type
estimator = args.estimator
k_prime = args.k_prime
t_partition = args.t_partition

IV_2SLS_DIR = f"output/rural_roads/{y_name}/"
OUTDIR = f"output/rural_roads/{y_name}/{RD_type}/{estimator}__k_{k_prime}__t_{t_partition}__isotropic/"

#########################################
# GP model specifications               #
# Note we use an anisotropic GP for     #
# real spatial data                     #
#########################################
torch.random.manual_seed(343165)

class ExactGPModelJustRBF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,ard=False):
        super(ExactGPModelJustRBF, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        if ard:
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
        else:
            self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TwoStageGPJustRBF(object):
    r"""A single-task exact GP model using a heteroskeastic noise model, with
    a Gaussian kernel and fixed noise likelihood
    """

    def __init__(self,train_x,train_y,observed_var):
  
        self.TE_likelihood = FixedNoiseGaussianLikelihood(
            noise=observed_var,
            learn_additional_noise=False
        )

        self.TE_model = ExactGPModelJustRBF(
            train_x=train_x, train_y=train_y.flatten(), likelihood=self.TE_likelihood
        )

        self.TE_model.train()
        self.TE_likelihood.train()

    def fit(self,train_x,train_y,observed_var):
        self.TE_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.TE_model.likelihood, self.TE_model)        
        self.TE_mll.train()
        fit_TE_model = botorch.fit.fit_gpytorch_model(self.TE_mll, optimizer=fit_gpytorch_torch,)
        self.TE_mll.eval()
        self.TE_model.eval()
        self.TE_likelihood.eval()

class TwoStageGPJustRBFWrapper(object):
    def fit(self,train_x,train_y,observed_var):
        
        self.model = TwoStageGPJustRBF(train_x,
                                       train_y,
                                       observed_var)
        
        # Stage one: Fit the noise model
        self.model.fit(train_x,train_y,observed_var)
    
    def posterior(self,test_x):
        with torch.no_grad():
            posterior = self.model.TE_model(test_x)
            lower_p, upper_p = posterior.confidence_region()
            sd = posterior.variance**(1/2)
            mu = posterior.mean
        to_plot = np.concatenate([test_x.numpy(),
                                  lower_p.numpy().reshape(-1,1),
                                  upper_p.numpy().reshape(-1,1),
                                  mu.detach().numpy().reshape(-1,1),
                                  sd.detach().numpy().reshape(-1,1)],axis=1)
        to_plot = pd.DataFrame(to_plot)
        to_plot.columns = [f"X{i}" for i in range(1,test_x.shape[1]+1)] + ['lower','upper','mu','sd']

        return to_plot

    def posterior_predictive(self,test_x):
        noise_posterior_mean = torch.clamp(torch.tensor(self.noise_posterior(test_x).mu.values),
                                           MIN_SE)
        with torch.no_grad():
            predictive_posterior = self.model.TE_likelihood(
                self.model.TE_model(test_x), noise=noise_posterior_mean.flatten()
            )
            lower_predictive, upper_predictive = predictive_posterior.confidence_region()
            
        to_plot = np.concatenate([test_x.numpy(),
                                  lower_predictive.numpy().reshape(-1,1),
                                  upper_predictive.numpy().reshape(-1,1)],axis=1)
        to_plot = pd.DataFrame(to_plot)
        to_plot.columns = [f"X{i}" for i in range(1,test_x.shape[1]+1)] + ['lower','upper','mu']

        return to_plot

    def describe(self):
        with torch.no_grad():
            description = {
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
    

def get_padded_posterior(models,target,train_x,test_x,random_test_sample,jitter=1e-4):
    # Use Rasmussmen eq. 2.42 to compute covariance of g*
    post = models[target].model.TE_model(test_x[random_test_sample,:])
    H = torch.ones(size=(1,train_x.shape[0]))
    Ky = models[target].model.TE_model.covar_module(train_x,train_x).add_diag(CATE_est_var)
    Kstar = models[target].model.TE_model.covar_module(train_x,test_x[random_test_sample,:]).evaluate()
    Hstar = torch.ones(size=(1,len(random_test_sample)))
    R = Hstar - (H @ Ky.inv_matmul(Kstar))
    add_cov = R.T @ torch.inverse(H @ Ky.inv_matmul(H.T)) @ R
    post = MultivariateNormal(mean=post.mean,
                              covariance_matrix=post.covariance_matrix + add_cov + torch.eye(len(random_test_sample))*jitter)
    return post
  
################################################
# Main inference exercise                      #
################################################

# Load the VKNN estimates
VKNN_ests = pd.read_csv(f'{OUTDIR}/roads_VKNN_ests.csv')
VKNN_ests['bias'] = VKNN_ests['obs_est'] - VKNN_ests['CATE']
train_x = torch.tensor(VKNN_ests[['longitude','latitude']].values, dtype=torch.float)
CATE_est_var = torch.tensor(VKNN_ests.ses.values, dtype=torch.float)**2


# Load the full dataset
full_data = pd.read_csv(f'{IV_2SLS_DIR}/roads_with_CATE.csv')
test_x = torch.from_numpy(full_data[['longitude','latitude']].values).type(torch.float)
cf_test_CATE_est = full_data['obs_est']

# Compute minimum distance to a training instance
d = pairwise_distances(test_x,train_x)
min_d = d.min(axis=1)

#########################################
# Consider four weighting strategies    #
#     - MLL weighting                   #
#     - LOO weighting                   #
#     - Weighted BLOOCV                 #
#     - 1-MC BLOOCV                     #
# But don't use weighted BLOOCV for     #
# unfiltered case                       #
#########################################

target_cols = ['bias','CATE']
results = {}
gp_test_posterior_means = {}

models = {}

for t in target_cols:
    print(f'Fitting {t} GP...')
    train_y = torch.from_numpy(VKNN_ests[t].values).type(torch.float)

    gp = TwoStageGPJustRBFWrapper()
    gp.fit(train_x,train_y,CATE_est_var)
    models[t] = gp
    lml = gp.get_lml(train_x,train_y,CATE_est_var)
    results[t] = {
        'MLL': lml
    }
    post_df = gp.posterior(test_x)
    gp_test_posterior_means[t] = post_df['mu']
    
gp_test_posterior_means['bias'] = cf_test_CATE_est - gp_test_posterior_means['bias']
    
#########################################
# Get the weighted estimators           #
#########################################

strategies = ['MLL']
for strategy in strategies:
    
    print(f'Computing {strategy} weighting MSE...')

    weights = scipy.special.softmax(
        np.array([results['bias'][strategy],results['CATE'][strategy]])
    )
    
    # Compute weighted mean
    posterior_weighted_mean = weights[0]*gp_test_posterior_means['bias'] +\
                              weights[1]*gp_test_posterior_means['CATE']
    
    gp_test_posterior_means[strategy] = posterior_weighted_mean

#########################################
# Compute local estimate for comparison #
#########################################

print('Computing local estimates...')
ests = {}

# First compute the estimates for our GP debiasing method
# Using a subset of N_TEST_SAMPLE villages drawn from the test set (to shorten runtimes)
N_TEST_SAMPLE=5000
random_test_sample = np.random.choice(test_x.shape[0],size=N_TEST_SAMPLE,replace=False)

with torch.no_grad():
    bias_ps = get_padded_posterior(models,'bias',train_x,test_x,random_test_sample)
    CATE_ps = get_padded_posterior(models,'CATE',train_x,test_x,random_test_sample)
    
bias_samples = np.zeros((N_TEST_SAMPLE,1000))
with torch.no_grad():
    for i in range(bias_samples.shape[1]):
        bias_sample = full_data.iloc[random_test_sample]['obs_est'] - \
                        bias_ps.rsample().detach().numpy()
        CATE_sample = CATE_ps.rsample().detach().numpy()
        mask = np.random.choice([0,1],size=N_TEST_SAMPLE,replace=True,p=weights)
        bias_samples[:,i] = bias_sample*mask + CATE_sample*(1-mask)


############################################
# Make two maps suggesting the spatial RDs #
############################################

print('Making maps...')
# Load indian state boundaries
#fname = 'data/rural_roads/India_States_ADM1_GADM-shp/3e563fd0-8ea1-43eb-8db7-3cf3e23174512020330-1-layr7d.ivha.shp'
fname = 'data/rural_roads/India_States_AMD1_GADM-shp/India_State_Boundary.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')

gp_test_posterior_means = pd.DataFrame(gp_test_posterior_means)
gp_test_posterior_means.columns = [f'posterior_{col}' for col in gp_test_posterior_means.columns]

merged = pd.concat([full_data,gp_test_posterior_means],axis=1)
 
raw_roads = pd.read_csv('output/rural_roads/roads.csv')
raw_long_lat = raw_roads[['longitude','latitude']]
raw_long_lat.columns = [f"{x}_raw" for x in raw_long_lat.columns]
merged = pd.concat([merged,raw_long_lat],axis=1)

merged.to_csv(f'{OUTDIR}/for_side_by_side_maps.csv')

# Map one shows local discontinuities in treatment across space

fig = plt.figure(figsize=(8,10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.title('Villages in sample')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax.set_extent([67, 90.5, 6, 40], ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, edgecolor='none')
ax.add_feature(shape_feature,facecolor='none')

plt.scatter(merged.loc[merged.r2012==0,'longitude_raw'],
            merged.loc[merged.r2012==0,'latitude_raw'],
            c='red',label='Untreated',
            transform=ccrs.PlateCarree(), s=0.5, marker='o')

plt.scatter(merged.loc[merged.r2012==1,'longitude_raw'],
            merged.loc[merged.r2012==1,'latitude_raw'],
            c='blue',label='Treated',
            transform=ccrs.PlateCarree(), s=0.5, marker='o')

ax.legend()
plt.savefig(f'{OUTDIR}/treatment_map.png')

# Map two shows spatial heterogeneity in TE estimates
# along with the quartile cutpoints

quartiles = merged.longitude_raw.quantile([0.25,0.5,0.75])
projection = ccrs.PlateCarree()
axes_class = (GeoAxes,
              dict(map_projection=projection))
fig = plt.figure(figsize=(8,10))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(1, 1),
                axes_pad=0.6,
                cbar_location='right',
                cbar_mode='single',
                cbar_pad=0.2,
                cbar_size='3%',
                label_mode='')  # note the empty label_mode

axgr[0].set_title(r'Villages in sample: estimated $\hat{\tau}(x)$')
axgr[0].set_extent([67, 90.5, 6, 40], ccrs.PlateCarree())
axgr[0].add_feature(cfeature.OCEAN)
axgr[0].add_feature(cfeature.LAND, edgecolor='none')
axgr[0].add_feature(shape_feature,facecolor='none')

sc = axgr[0].scatter(merged.longitude_raw,
                     merged.latitude_raw,
                     c=merged.posterior_MLL.values,
                     transform=ccrs.PlateCarree(), s=0.5, marker='o')

axgr[0].scatter(VKNN_ests.longitude_raw,
                VKNN_ests.latitude_raw,
                c='red',label='VKNN center',
                transform=ccrs.PlateCarree(),
                s=10, marker='o')
axgr.cbar_axes[0].colorbar(sc)

plt.savefig(f'{OUTDIR}/TE_map.png')

####################################################################################
# Make errorbar plots for our local estimates vs. the subset IVREG estimates       #
####################################################################################
print('Making error bar plots...')
# Load the state labels
all_raw_data = pd.read_csv('output/rural_roads/roads_w_controls.csv')
merged['state'] = all_raw_data['pc01_state_name']

# Get mean TE estimates across states
region_group = merged.state.str.title()
bias_samples_df = pd.DataFrame(bias_samples)
bias_samples_region_group = region_group.iloc[random_test_sample].reset_index(drop=True)

GP_regional_means = bias_samples_df.groupby(bias_samples_region_group).mean().mean(axis=1)
GP_regional_SEs = (bias_samples_df.groupby(bias_samples_region_group).mean().var(axis=1))**(1/2)
full_s = {'Estimate':bias_samples_df.mean().mean(),'Std. Error':(bias_samples_df.mean().var())**(1/2)}

our_regional_estimates = pd.concat([GP_regional_means.rename('Estimate'),
                                    GP_regional_SEs.rename('Std. Error')],
                                   axis=1)

our_regional_estimates = pd.concat([our_regional_estimates,pd.Series(full_s).rename('All states').to_frame().T],axis=0)
our_regional_estimates.index.name = 'State'

# Load the subset IVREG estimates
local_ests = pd.read_csv(f'{IV_2SLS_DIR}/quartile_ivreg_results.csv')
local_ests['State'] = local_ests['State'].str.replace('.',' ',regex=False)
local_ests.set_index('State',inplace=True)

# Make merged TE table
merged_ests = pd.merge(
    (local_ests.Estimate.round(3).astype(str) + ' (' + local_ests['Std. Error'].round(3).astype(str) + ')').rename('Asher & Novosad RDD subset analysis'),
    (our_regional_estimates.Estimate.round(3).astype(str) + ' (' + our_regional_estimates['Std. Error'].round(3).astype(str) + ')').rename('GP debiasing'),
    left_index=True,right_index=True
)

merged_ests.style.to_latex(f'{OUTDIR}/quartile_spatial_hetero.tex')
merged_ests.to_excel(f'{OUTDIR}/quartile_spatial_hetero.xlsx')

# Make error bar plot
to_plot = pd.concat([local_ests.assign(Est='Asher & Novosad RDD'),
                     our_regional_estimates.assign(Est='GP debiasing')],
                    axis=0).reset_index()

to_plot.to_csv(f'{OUTDIR}/full_and_state_specific_TE_estimates.csv')

fig,ax = plt.subplots(figsize=(9,6))
ax = sns.pointplot(data=to_plot, x ='State', y='Estimate', hue='Est',
                   dodge=True, join=False, errorbar=None)

# Find the x,y coordinates for each point
x_coords = []
y_coords = []
for point_pair in ax.collections:
    for x, y in point_pair.get_offsets():
        x_coords.append(x)
        y_coords.append(y)

errors = to_plot['Std. Error']
colors = ['steelblue']*len(local_ests) + ['coral']*len(local_ests)
ax.errorbar(x_coords, y_coords, yerr=errors*1.96,
            ecolor=colors, fmt=' ', zorder=-1)

ax.axhline(0,linestyle='--',color='grey')

ax.set_xlabel('State')
ax.set_ylabel(r'$\hat{\tau} \pm 1.96 SE_{\hat{\tau}}$')
ax.set_title(r'$\hat{\tau}$ by state')
plt.xticks(rotation = 45)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUTDIR}/quartile_spatial_hetero.png')
