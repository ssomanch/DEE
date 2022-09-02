import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

xs = ['longitude','latitude']
z = 't'
t = 'r2012'
ys_regex = 'ndex_andrsn$'
linear_controls = ['left','right']
controls = ['primary_school','med_center','elect','tdist',
            'irr_share','ln_land','pc01_lit_share',
            'pc01_sc_share','bpl_landed_share','bpl_inc_source_sub_share',
            'bpl_inc_250plus','vhg_dist_id']

# Load the raw data
df = pd.read_stata('data/rural_roads/pmgsy_working_aer.dta')

# Apply the sample filter
df = df.loc[(df.secc_pop_ratio>=0.8) & (df.secc_pop_ratio <= 1.2)]
df = df.loc[(df.app_pr == 0) | (df.con00 == 0)]
restricted_data = pd.read_stata('data/rural_roads/pmgsy_working_aer_mainsample.dta')
df = df.loc[df['pc01_state_name'].isin(restricted_data['pc01_state_name'].unique())]

# Identify the outcome columns
ys = df.filter(regex=ys_regex,axis=1).columns.to_list()

# Rename total population
df['total_pop'] = df['pc01_pca_tot_p']

# Identify full set of columns needed and restrict to complete cases
all_cols = xs + ['total_pop',z,t] + linear_controls + ys + controls
missing_vals = df[all_cols].isnull().any(axis=1)
has_all = df.loc[~missing_vals]

# Write out the minimal geo + pop DF for LORD3
has_all[xs + ['total_pop',t] + ys].to_csv('output/rural_roads/roads.csv',index=False)

has_all[['longitude','pc01_state_name',z,t] + ys + linear_controls + controls].to_csv('output/rural_roads/roads_w_controls.csv',index=False)