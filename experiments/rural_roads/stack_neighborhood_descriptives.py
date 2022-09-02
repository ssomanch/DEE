import pandas as pd

pop_p1 = pd.read_csv('output/rural_roads/transport_index_andrsn/population/nonparametric_estimator__k_400__t_40__isotropic/all_village_Ns.csv',index_col=0)
pop_p2 = pd.read_csv('output/rural_roads/transport_index_andrsn/population/nonparametric_estimator__k_400__t_40__isotropic/all_RD_bws.csv',index_col=0)
ps_p1 = pd.read_csv('output/rural_roads/transport_index_andrsn/space_and_population/nonparametric_estimator__k_400__t_40__isotropic/all_village_Ns.csv',index_col=0)
ps_p2 = pd.read_csv('output/rural_roads/transport_index_andrsn/space_and_population/nonparametric_estimator__k_400__t_40__isotropic/all_RD_bws.csv',index_col=0)

pop_p1.rename({'All RDs': 'All'},axis=1,inplace=True)
pop_p2.rename({'All RDs': 'All'},axis=1,inplace=True)
ps_p1.rename({'All RDs': 'All'},axis=1,inplace=True)
ps_p2.rename({'All RDs': 'All'},axis=1,inplace=True)

pop_p1 = pd.concat([pop_p1[['All']]], keys=['Population RDs'], names=[''],axis=1)
pop_p1.loc[['$\mathcal{L}$','Validated $\mathcal{L}$']] = '--'
ps_p1 = pd.concat([ps_p1], keys=['Population and Spatial RDs'], names=[''],axis=1)

pop_p2 = pd.concat([pop_p2[['All']]], keys=['Population RDs'], names=[''],axis=1)
ps_p2 = pd.concat([ps_p2], keys=['Population and Spatial RDs'], names=[''],axis=1)

p1 = pd.concat([ps_p1,pop_p1],axis=1)
p2 = pd.concat([ps_p2,pop_p2],axis=1)

p1 = pd.concat([p1], keys=['N RDs'], names=[''],axis=0)
p2 = pd.concat([p2], keys=['VK[D]'], names=[''],axis=0)

full_panel = pd.concat([p1,p2],axis=0)

full_panel[['Population and Spatial RDs','Population RDs']].to_latex(
  'output/rural_roads/figures/VKNN_descriptives.tex',
  multirow=True,multicolumn_format='c|',column_format='ll|lll|l',escape=False
)