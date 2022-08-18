import torch
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate X and D for initial computation of Voroni x KNN centers')
parser.add_argument('--seed1', help='Seed forsampling',type=int)

args = parser.parse_args()

np.random.seed(args.seed1)
torch.manual_seed(args.seed1)

N = 20000
p = 2

# Need to init with X, Y, but will just sample in prior mode
Y = torch.from_numpy(np.ones(N))

# Uniformly sample X
X = torch.tensor(np.random.rand(N,p), dtype=torch.float)

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
