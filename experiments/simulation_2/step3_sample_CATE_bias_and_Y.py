import torch
import numpy as np
import pandas as pd
import argparse

from data_generator import DataGenerator,get_bias_multiplier

import dill

import argparse

parser = argparse.ArgumentParser(description='Compute the true CATE and E[Bias] at Voroni centers')
parser.add_argument('--seed1', help='Seed for dataset sampling',type=int)
parser.add_argument('--CATE-ls', help='Lengthscale for CATE GP prior',type=float)
parser.add_argument('--bias-ls', help='Lengthscale for bias GP prior',type=float)

args = parser.parse_args()

OUTDIR = f'output/simulation_2/{args.seed1}/{args.CATE_ls}/{args.bias_ls}/'

np.random.seed(args.seed1)
torch.manual_seed(args.seed1)

# Load the problem instance
with open(f'{OUTDIR}/problem_instance.pkl','rb') as f:
    problem_instance = dill.loads(f.read())

# Load the oracle centers
LORD3_df = pd.read_csv(f'{OUTDIR}/TE_estimates_at_voroni_KNN_centers_without_true_parameters.csv')
LORD3_Xs = torch.tensor(LORD3_df[['X1','X2']].values, dtype=torch.float)

# Compute the CATE and bias
LORD3_CATE, LORD3_bias = problem_instance.compute_CATE_and_bias_at_X(LORD3_Xs)

pd.DataFrame({'CATE':LORD3_CATE,
              'bias':LORD3_bias})\
  .to_csv(f'{OUTDIR}/true_CATE_and_bias_at_voroni_KNN_centers.csv',index=False)
