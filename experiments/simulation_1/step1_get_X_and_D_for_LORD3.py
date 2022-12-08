import torch
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate X and D for initial computation of Voroni x KNN centers')
parser.add_argument('--seed1', help='Seed forsampling',type=int)
parser.add_argument('--sample-at-RDD', help = 'Generate more samples at the RDD', type=int, default=0)

args = parser.parse_args()

np.random.seed(args.seed1)
torch.manual_seed(args.seed1)

N = 20000
p = 2

# Need to init with X, Y, but will just sample in prior mode
Y = torch.from_numpy(np.ones(N))

if args.sample_at_RDD == 0:
    # Uniformly sample X
    X = torch.tensor(np.random.rand(N,p), dtype=torch.float)
else:
    # In this is special case in simulation 1, we increase the number of samples 
    # around the discontinuity to increase the number VK centers (|U|) discovered.   
    # This increase in the VK centers discovered translates to better MSE performance 
    # of BLOOCV methods as compared to marginal or LOOCV log likelihood. 
    # This increase in sampling is discontinuities achieves more VK centers without 
    # increasing the total number of samples (which makes GP based simulation hard in step 3).
    uniform_points = np.random.rand(N//2, p)
    rest_points_size = N//2
    # Generate rest_points_size points around discontinuity in the 2-D space. 
    epsilon = 0.01
    rest_points_1 = np.vstack((np.random.uniform(low=0.25-epsilon, high=0.25+epsilon, size=rest_points_size//4),
                             np.random.uniform(low=0.25-epsilon, high=1, size=rest_points_size//4))).T
    rest_points_2 = np.vstack((np.random.uniform(low=0.25-epsilon, high=1, size=rest_points_size//4),
                               np.random.uniform(low=0.25-epsilon, high=0.25+epsilon, size=rest_points_size//4))).T
    rest_points_3 = np.vstack((np.random.uniform(low=0.65-epsilon, high=0.65+epsilon, size=rest_points_size//4),
                             np.random.uniform(low=0.65-epsilon, high=1, size=rest_points_size//4))).T
    rest_points_4 = np.vstack((np.random.uniform(low=0.65-epsilon, high=1, size=rest_points_size//4),
                               np.random.uniform(low=0.65-epsilon, high=0.65+epsilon, size=rest_points_size//4))).T
    rest_points = np.concatenate([rest_points_1, rest_points_2, rest_points_3, rest_points_4])
    
    X = torch.tensor(np.concatenate([uniform_points, rest_points]), dtype=torch.float)

# Randomly sample G
G = np.random.choice(['a','c1','c2','n'],size=N,p=[0.1,0.4,0.4,0.1])

# Compute RD instruments
b1 = torch.from_numpy(np.array([0.65,0.65]))
b2 = torch.from_numpy(np.array([0.25,0.25]))
Z2 = ((X > b1).sum(axis=1)==2)
Z1 = ((X > b2).sum(axis=1)==2)

# Compute observed treatment
D = torch.from_numpy(G=='a')*Z2 + \
    torch.from_numpy((G=='a') | (G=='c1'))*Z1*(~Z2) + \
    torch.from_numpy((G=='a') | (G=='c1') | (G=='c2'))*(~Z1)*(~Z2)

# Write out csv so we can compute Voroni/KNN centers
df = pd.DataFrame(X.detach().numpy())
df.columns = ['X1','X2']
df['Y'] = Y.detach().numpy()
df['D'] = D.detach().numpy()
df['G'] = G
df['Z1'] = Z1
df['Z2'] = Z2

df.to_csv(f'output/simulation_1/{args.seed1}/LORD3_inputs.csv',index=False)
