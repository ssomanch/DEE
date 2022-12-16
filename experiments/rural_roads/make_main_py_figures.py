import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Plotting libraries

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from mpl_toolkits.axes_grid1 import AxesGrid

sns.set_context('paper',font_scale=2)

#######################################################
# Parameters #
#######################################################

estimator = 'nonparametric_estimator'
k = 400
t = 40
# top_M is chosen based on randomization testing by running 
# randomization_testing_LORD3.R and  compute_M_prime_from_randomization.R
# Note that these files are not run in the regular flow of the application.
top_M=2784

#######################################################
# Utilities                                           #
#######################################################

def get_table_and_figure(estimator,k,t):
    """Load all of the estimates using this estimator and VKNN parameterization.
    
    Returns
    -------
        fig : plt.Figure showing the estimates
        ATEs : pd.DataFrame with all estimates + SEs
        ATE_pivot_str : str table with tau (SE) for latex
    """
    Y_map = {
        'agriculture_index_andrsn': 'Ag production',
        'consumption_index_andrsn': 'Consumption',
        'firms_index_andrsn': 'Firms',
        'occupation_index_andrsn': 'Occupation',
        'transport_index_andrsn': 'Transportation'
    }
    
    ###################################
    # Load data                       #
    ###################################
    
    target_TE_ests = glob.glob(f'output/rural_roads/**/{estimator}__k_{k}__t_{t}__isotropic/full_and_state_specific_TE_estimates.csv',recursive=True)
    index_and_RDDs = pd.DataFrame([x.split('/') for x in target_TE_ests]).iloc[:,[2,3]]
    index_and_RDDs['path'] = target_TE_ests
    all_dfs = []
    for i,r in index_and_RDDs.iterrows():
        all_dfs.append(pd.read_csv(r['path'],index_col=0).assign(Y=r[2],RDDs=r[3]))
        
    all_dfs = pd.concat(all_dfs,axis=0)
    all_dfs = all_dfs.loc[~((all_dfs.Est=='Asher & Novosad RDD') & (all_dfs.RDDs=='space_and_population'))]
    ###################################
    # Subset to all states            #
    ###################################
    
    ATEs = all_dfs.loc[all_dfs.State=='All states']
    
    ATEs = ATEs.sort_values(['Y','Est','RDDs'])
    ATEs['Y'] = ATEs['Y'].map(Y_map)
    
    ATEs['RDDs'] = ATEs['RDDs'].map({'population':'GP debiasing, discovered pop. RDs',
                                     'space_and_population':'GP debiasing, discovered spatial and pop. RDs'})
    
    ATEs['Method'] = ATEs.apply(lambda r: r['RDDs'] if r['Est']=='GP debiasing' else r['Est'],axis=1)
    
    ATE_pivot = pd.pivot(ATEs,index='Y',columns='Method')[['Estimate','Std. Error']]
    ATE_pivot_str = ATE_pivot['Estimate'].round(3).astype(str) + ' (' + ATE_pivot['Std. Error'].round(3).astype(str) + ")"
    
    ATE_pivot_str.index.name='Outcome'
    ATE_pivot_str = ATE_pivot_str[['Asher & Novosad RDD','GP debiasing, discovered pop. RDs','GP debiasing, discovered spatial and pop. RDs']]
    
    ###################################
    # Make Figure                     #
    ###################################
    
    ATEs = ATEs.sort_values(['Y','Method'])
    
    fig,ax = plt.subplots(figsize=(12,6))
    
    n_methods = ATEs.Method.nunique()

    rgb_values = sns.color_palette("Set2", n_methods)
    palette = sns.color_palette("deep", n_methods)
    ax = sns.pointplot(data=ATEs, x = 'Y', y = 'Estimate', hue='Method', palette=palette,
                       dodge=True, join=False, errorbar=None)


    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)


    errors = ATEs.sort_values(['Method','Y'])['Std. Error']
    ax.errorbar(x_coords, y_coords, yerr=errors*1.96,
                ecolor=np.repeat(palette,repeats=ATEs['Y'].nunique(),axis=0), fmt=' ', zorder=-1)

    ax.axhline(0,linestyle='--',color='grey')
    ax.legend(loc='upper left',fontsize=12)
    ax.set_xlabel('Outcome index')
    ax.set_ylabel(r'Estimate ($E[\hat{\tau}(x)]$) and 95% CI')
    ax.set_title(r'Comparing $E[\hat{\tau}(x)]$ across three methods')
    
    return ATEs, ATE_pivot_str, fig
  
def make_side_by_side_maps(y_name,ATE_pivot_str,estimator,k_prime,t_partition):
    
    
    Y_map = {
        'agriculture_index_andrsn': 'Ag production',
        'consumption_index_andrsn': 'Consumption',
        'firms_index_andrsn': 'Firms',
        'occupation_index_andrsn': 'Occupation',
        'transport_index_andrsn': 'Transportation'
    }
    
    vmin,vmax = get_vmax_min(y_name,estimator,k_prime,t_partition)
    vmin = -max(abs(vmin),abs(vmax))
    vmax = max(abs(vmin),abs(vmax))
    # Map two shows spatial heterogeneity in TE estimates
    # along with the quartile cutpoints
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    fig = plt.figure(figsize=(20,15))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(1, 2),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=1,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode

    axgr, sc = add_one_map(y_name,'population',axgr,0,vmin,vmax,ATE_pivot_str.loc[Y_map[y_name]],estimator,k_prime,t_partition)
    axgr, sc = add_one_map(y_name,'space_and_population',axgr,1,vmin,vmax,ATE_pivot_str.loc[Y_map[y_name]],estimator,k_prime,t_partition)
    
    axgr.cbar_axes[0].colorbar(sc)
    
    
    
    plt.suptitle(r'GP extrapolated $\hat{\tau}(x)$ estimates for ' + Y_map[y_name].title() + ' outcome',fontsize=24,y=0.92)
    
    return fig

def get_vmax_min(y_name,estimator,k_prime,t_partition):
    vmin,vmax=1000,-1000
    for RD_type in ['population','space_and_population']:
        OUTDIR = f"output/rural_roads/{y_name}/{RD_type}/{estimator}__k_{k_prime}__t_{t_partition}__isotropic/"
        merged = pd.read_csv(f'{OUTDIR}/for_side_by_side_maps.csv')
        vmin,vmax= min(vmin,merged['posterior_min dist'].min()),max(vmax,merged['posterior_min dist'].max())
    return vmin,vmax
    
def add_one_map(y_name,RD_type,axgr,i,vmin,vmax,table,
                estimator,k_prime,t_partition):
    # Load the VKNN estimates & GP debiasing estimates
    OUTDIR = f"output/rural_roads/{y_name}/{RD_type}/{estimator}__k_{k_prime}__t_{t_partition}__isotropic/"
    VKNN_ests = pd.read_csv(f'{OUTDIR}/roads_VKNN_ests.csv')
    merged = pd.read_csv(f'{OUTDIR}/for_side_by_side_maps.csv')

    if RD_type == 'population':
        ate = table['GP debiasing, discovered pop. RDs']
    else:
        ate = table['GP debiasing, discovered spatial and pop. RDs']
    axgr[i].set_title(r'Using discovered ' + f"{RD_type.replace('_',' ').replace('space','spatial')} RDs\n" +\
                      f"(ATE = {ate})")
    axgr[i].set_extent([67, 90.5, 6, 40], ccrs.PlateCarree())
    axgr[i].add_feature(cfeature.OCEAN)
    axgr[i].add_feature(cfeature.LAND, edgecolor='none')
    axgr[i].add_feature(shape_feature,facecolor='none',edgecolor='grey')

    sc = axgr[i].scatter(merged.longitude_raw,
                         merged.latitude_raw,
                         c=merged['posterior_min dist'].values,
                         transform=ccrs.PlateCarree(),
                         vmin=vmin,vmax=vmax,cmap='coolwarm_r',
                         s=0.5, marker='o')

    axgr[i].scatter(VKNN_ests.longitude_raw,
                    VKNN_ests.latitude_raw,
                    c='black',label='VKNN center',
                    transform=ccrs.PlateCarree(),
                    s=10, marker='o')
    return axgr, sc

#######################################################
# Figure 1: ATE across all outcomes                   #
#######################################################

ys = [
    f"{y}_index_andrsn" for y in ['transport','occupation','firms','consumption','agriculture']
]

ATEs, ATE_pivot_str, fig = get_table_and_figure(estimator,k,t)
ATE_pivot_str.columns.name = 'Estimation strategy'
ATE_pivot_str.to_latex('output/rural_roads/figures/ATE_by_outcomes.tex')
plt.savefig('output/rural_roads/figures/ATE_by_outcomes.png')


#######################################################
# Figure 2: Sample treatment + L.                     #
#######################################################

# Load indian state boundaries
fname = 'data/rural_roads/India_States_AMD1_GADM-shp/India_State_Boundary.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')

# Make presentation grade treatment map
y_name = 'transport_index_andrsn'
RD_type = 'space_and_population'
k_prime = k
t_partition = t

OUTDIR = f"output/rural_roads/{y_name}/{RD_type}/{estimator}__k_{k_prime}__t_{t_partition}__isotropic/"
merged = pd.read_csv(f'{OUTDIR}/for_side_by_side_maps.csv')

projection = ccrs.PlateCarree()
axes_class = (GeoAxes,
              dict(map_projection=projection))
fig = plt.figure(figsize=(20,15))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(1, 2),
                axes_pad=0.6,
                label_mode='')  # note the empty label_mode

ax = axgr[0]

ax.set_title('Villages in sample',fontsize=24)
ax.set_extent([67, 90.5, 6, 40])
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, edgecolor='none')
ax.add_feature(shape_feature,facecolor='none',edgecolor='grey')


ax.scatter(merged['longitude_raw'],
           merged['latitude_raw'],
           c=merged.r2012.map({0:'red',1:'blue'}),
           transform=ccrs.PlateCarree(), s=2,alpha=0.5, marker='o')

# Add two duplicate points to hack the legend

r1 = merged.loc[merged.r2012==1].sample(1)
ax.scatter(r1['longitude_raw'],
           r1['latitude_raw'],
           c=r1.r2012.map({0:'red',1:'blue'}), label='Treated',
           transform=ccrs.PlateCarree(), s=2,alpha=0.5, marker='o')
r0 = merged.loc[merged.r2012==0].sample(1)
ax.scatter(r0['longitude_raw'],
           r0['latitude_raw'],
           c=r0.r2012.map({0:'red',1:'blue'}), label='Untreated',
           transform=ccrs.PlateCarree(), s=2,alpha=0.5, marker='o')

lgnd = ax.legend()

#change the marker size manually for both lines
lgnd.legendHandles[0].set_sizes([18])
lgnd.legendHandles[1].set_sizes([18])

ax = axgr[1]

ax.set_title('Validated RDs included in ' + r'$\mathcal{L}$ by type',fontsize=24)
ax.set_extent([67, 90.5, 6, 40])
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, edgecolor='none')
ax.add_feature(shape_feature,facecolor='none',edgecolor='grey')

valid_status = pd.read_csv('output/rural_roads/roads_LORD3_w_RDD_category_and_p_val.csv')
raw_roads = pd.read_csv('output/rural_roads/roads.csv')
valid_status[['longitude','latitude']] = raw_roads[['longitude','latitude']]
top_M = valid_status.loc[(valid_status.top_M_prime) & (valid_status.valid_RD)]
top_M['RD_type'] = top_M['total_pop_RD'].map({True:'Population',False:'Spatial'})
print(top_M.shape)
for t in top_M.RD_type.unique():
    ax.scatter(top_M.loc[top_M['RD_type']==t,'longitude'],
                top_M.loc[top_M['RD_type']==t,'latitude'],label=t,
                transform=ccrs.PlateCarree(), s=3, marker='o')

lgnd = ax.legend()

#change the marker size manually for both lines
lgnd.legendHandles[0].set_sizes([18])
lgnd.legendHandles[1].set_sizes([18])


plt.savefig('output/rural_roads/figures/villages_and_L.png')

#######################################################
# Separate villages by treatment map       #
#######################################################

projection = ccrs.PlateCarree()
axes_class = (GeoAxes,
              dict(map_projection=projection))
fig = plt.figure(figsize=(10,15))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(1, 1),
                axes_pad=0.6,
                label_mode='')  # note the empty label_mode

ax = axgr[0]

ax.set_title('Villages in sample',fontsize=24)
ax.set_extent([67, 90.5, 6, 40])
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, edgecolor='none')
ax.add_feature(shape_feature,facecolor='none',edgecolor='grey')


ax.scatter(merged['longitude_raw'],
           merged['latitude_raw'],
           c=merged.r2012.map({0:'red',1:'blue'}),
           transform=ccrs.PlateCarree(), s=2,alpha=0.5, marker='o')

# Add two duplicate points to hack the legend

r1 = merged.loc[merged.r2012==1].sample(1)
ax.scatter(r1['longitude_raw'],
           r1['latitude_raw'],
           c=r1.r2012.map({0:'red',1:'blue'}), label='Treated',
           transform=ccrs.PlateCarree(), s=2,alpha=0.5, marker='o')
r0 = merged.loc[merged.r2012==0].sample(1)
ax.scatter(r0['longitude_raw'],
           r0['latitude_raw'],
           c=r0.r2012.map({0:'red',1:'blue'}), label='Untreated',
           transform=ccrs.PlateCarree(), s=2,alpha=0.5, marker='o')

lgnd = ax.legend()

#change the marker size manually for both lines
lgnd.legendHandles[0].set_sizes([18])
lgnd.legendHandles[1].set_sizes([18])

plt.savefig('output/rural_roads/figures/villages_by_T.png')


#######################################################
# Separate RD by RD type map                          #
#######################################################

projection = ccrs.PlateCarree()
axes_class = (GeoAxes,
              dict(map_projection=projection))
fig = plt.figure(figsize=(10,15))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(1, 1),
                axes_pad=0.6,
                label_mode='')  # note the empty label_mode

ax = axgr[0]

ax.set_title('Validated RDs included in ' + r'$\mathcal{L}$ by type',fontsize=24)
ax.set_extent([67, 90.5, 6, 40])
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, edgecolor='none')
ax.add_feature(shape_feature,facecolor='none',edgecolor='grey')

valid_status = pd.read_csv('output/rural_roads/roads_LORD3_w_RDD_category_and_p_val.csv')
raw_roads = pd.read_csv('output/rural_roads/roads.csv')
valid_status[['longitude','latitude']] = raw_roads[['longitude','latitude']]
top_M = valid_status.loc[(valid_status.top_M_prime) & (valid_status.valid_RD)]
top_M['RD_type'] = top_M['total_pop_RD'].map({True:'Population',False:'Spatial'})
print(top_M.shape)
for t in top_M.RD_type.unique():
    ax.scatter(top_M.loc[top_M['RD_type']==t,'longitude'],
                top_M.loc[top_M['RD_type']==t,'latitude'],label=t,
                transform=ccrs.PlateCarree(), s=3, marker='o')

lgnd = ax.legend()

#change the marker size manually for both lines
lgnd.legendHandles[0].set_sizes([18])
lgnd.legendHandles[1].set_sizes([18])


plt.savefig('output/rural_roads/figures/validated_RDs_in_L.png')

#######################################################
# Figure 3-7: GP TE estimates for each outcome.       #
#######################################################

for y in ys:
  fig = make_side_by_side_maps(y,ATE_pivot_str,estimator,k_prime,t_partition)
  plt.savefig(f'output/rural_roads/figures/{y}_side_by_side.png',)
