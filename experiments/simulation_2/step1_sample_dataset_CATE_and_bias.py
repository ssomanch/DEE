import torch
import numpy as np
import pandas as pd
import argparse

from scipy.stats.distributions import norm

from src.data_generator import DataGenerator,get_bias_multiplier

import dill

parser = argparse.ArgumentParser(description='Sample dataset D')
parser.add_argument('--seed1', help='Seed for dataset sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)

args = parser.parse_args()

###############################################################
# Generate problem instance                                   #
###############################################################

OUTDIR = f'../output/simulation_2/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

np.random.seed(args.seed1)
torch.manual_seed(args.seed1)

N = 20000
p = 2
max_abs_rho = 0.9

gamma0 = norm.ppf(0.1)
gamma1 = norm.ppf(0.5)
gamma2 = norm.ppf(0.9)

scale_factor = np.mean([get_bias_multiplier(0,gamma0,gamma1,gamma2),
                        get_bias_multiplier(1,gamma0,gamma1,gamma2),
                        get_bias_multiplier(1,gamma0,gamma1,gamma2),
                        get_bias_multiplier(2,gamma0,gamma1,gamma2)])

CATE_hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1e-3),
    'covar_module.base_kernel.lengthscale': torch.tensor(float(args.CATE_ls)),
    'covar_module.outputscale': torch.tensor((scale_factor**2)/4.),
}
rhosigma_hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1e-3),
    'covar_module.base_kernel.lengthscale': torch.tensor(float(args.bias_ls)),
    'covar_module.outputscale': torch.tensor(1./4.),
}

problem_instance = DataGenerator(N,p,gamma0,gamma1,gamma2,max_abs_rho,CATE_hypers,rhosigma_hypers)


problem_instance.draw_CATE_and_rhosigma_f()
    
X,U,D,Z,Y0,Y1,Y,bias,tau = problem_instance.sample_train_data()

test_x1,test_x2 = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
test_x = np.concatenate([test_x1.flatten().reshape(1,-1).T,
                         test_x2.flatten().reshape(1,-1).T],
                       axis=1)
test_x = torch.tensor(test_x, dtype=torch.float)

test_CATE, test_bias = problem_instance.compute_CATE_and_bias_at_X(test_x)

all_train = np.concatenate([X,U,D[..., np.newaxis],Z[..., np.newaxis],
                            Y0[..., np.newaxis],Y1[..., np.newaxis],Y[..., np.newaxis],
                            bias[..., np.newaxis],tau[..., np.newaxis]],1)
all_train = pd.DataFrame(all_train)
all_train.columns = ['X1','X2','Ud','U0','U1','D','Z','Y0','Y1','Y','bias','tau']

###############################################################
# Write output                                                #
###############################################################

all_train.to_csv(f'{OUTDIR}/LORD3_inputs.csv',index=False)

pd.DataFrame({'CATE':test_CATE,
              'bias':test_bias})\
  .to_csv(f'{OUTDIR}/true_test_CATE_and_bias.csv',index=False)

with open(f'{OUTDIR}/problem_instance.pkl', "wb") as f:
    f.write(dill.dumps(problem_instance))