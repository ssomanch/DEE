import torch
import numpy as np
import pandas as pd
import gpytorch

from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

import argparse

import gc
import copy


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def generate_GPs(args, X, Y, all_Xs):
    
    ########################################
    # Sample the CATE                      #
    ########################################

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.),
        'covar_module.base_kernel.lengthscale': torch.tensor(float(args.CATE_ls)),
        'covar_module.outputscale': torch.tensor(5.),
    }

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    gp = ExactGPModel(X,Y,likelihood)
    gp.initialize(**hypers)

    gp.eval()
    likelihood.eval()

    with gpytorch.settings.prior_mode(True):
        prior_sample = gp(all_Xs)

    rsample = prior_sample.rsample()

    rsample_numpy = copy.deepcopy(rsample.detach().numpy())
   
    del gp
    del likelihood
    del rsample
    del prior_sample

    gc.collect(generation=2)

    ########################################
    # Sample the bias                      #
    ########################################

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.),
        'covar_module.base_kernel.lengthscale': torch.tensor(float(args.bias_ls)),
        'covar_module.outputscale': torch.tensor(5.),
    }

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = ExactGPModel(X,Y,likelihood)
    gp.initialize(**hypers)

    gp.eval()
    likelihood.eval()

    with gpytorch.settings.prior_mode(True):
        prior_sample = gp(all_Xs)

    rsample_bias = prior_sample.rsample()
    rsample_bias_numpy = copy.deepcopy(rsample_bias.detach().numpy())
    
    return rsample_numpy, rsample_bias_numpy

parser = argparse.ArgumentParser(description='Sample CATE, bias and Y')
parser.add_argument('--seed1', help='Seed for sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)
parser.add_argument('--root-dir', help='Path to root directory which contains output',type=str, default='')

args = parser.parse_args()

root_dir = args.root_dir

OUTDIR = f'{root_dir}output/simulation_1/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

np.random.seed(args.seed1)
torch.manual_seed(args.seed1)

# Load dataframe with training inputs
df = pd.read_csv(f'output/simulation_1/{args.seed1}/LORD3_inputs.csv')
N = df.shape[0]
Y = torch.from_numpy(df.Y.values)        # Random Y draw, to init GP priors
X = torch.from_numpy(df[['X1','X2']].values)
D = torch.from_numpy(df.D.values)
G = df.G.values

# Generate test grid
test_x1,test_x2 = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
test_x = np.concatenate([test_x1.flatten().reshape(1,-1).T,
                         test_x2.flatten().reshape(1,-1).T],
                       axis=1)
test_x = torch.from_numpy(test_x)

# Load the LORD3 repaired neighborhood centers
U = pd.read_csv(f'output/simulation_1/{args.seed1}/voroni_KNN_centers__ignore_TE_estimates.csv')
U_x = torch.from_numpy(U[['X1','X2']].values)

# Stack the training instances, test grid, and Voroni centers
all_Xs = torch.cat([X,test_x,U_x],0)
all_Xs = all_Xs.type(torch.FloatTensor)

rsample_numpy, rsample_bias_numpy = generate_GPs(args, X, Y, all_Xs)

train_CATE = rsample_numpy[:N,]
test_CATE = rsample_numpy[N:(N+test_x.shape[0]),]
U_CATE = rsample_numpy[(N+test_x.shape[0]):,]

train_bias = rsample_bias_numpy[:N,]
test_bias = rsample_bias_numpy[N:(N+test_x.shape[0]),]
U_bias = rsample_bias_numpy[(N+test_x.shape[0]):,]


pd.DataFrame({'CATE':U_CATE,
              'bias':U_bias})\
  .to_csv(f'{OUTDIR}/true_CATE_and_bias_at_voroni_KNN_centers.csv',index=False)

pd.DataFrame({'CATE':test_CATE,
              'bias':test_bias})\
  .to_csv(f'{OUTDIR}/true_test_CATE_and_bias.csv',index=False)

########################################
# Now solve linear system to get       #
# expected values of the additive      #
# functions on each complier type, for #
# the training data                    #
########################################

a = train_bias/2
c1 = 0*train_bias
c2 = -4/5*train_bias
n = -13/10*train_bias

Y0 = torch.from_numpy(-1/2*train_CATE)
Y1 = torch.from_numpy(1/2*train_CATE)

group_shift = (G=='a')*a + \
              (G=='c1')*c1 + \
              (G=='c2')*c2 + \
              (G=='n')*n

Y = Y1*D + Y0*(~D) + torch.from_numpy(group_shift)
Y = Y + torch.randn_like(Y)         # Assuming standard normal outcome noise

# Rewrite out csv for downstream
df['Y'] = Y.detach().numpy()
df['CATE'] = train_CATE
df['bias'] = train_bias

df.to_csv(f'{OUTDIR}/LORD3_inputs_and_CATE_bias_Y.csv',index=False)