library(data.table)
library(AER)
library(ivpack)
library(stringr)

# Parse target Y
args = commandArgs(trailingOnly=TRUE)
y_name = args[1]
OUTDIR = paste0('output/rural_roads/',y_name,'/')
dir.create(OUTDIR, showWarnings = FALSE)

raw_roads = fread('output/rural_roads/roads_w_controls.csv')
raw_roads[,z:=left+right]
# Use same bandwidth for now...
raw_roads = raw_roads[abs(z)<=83,]

# Drop NAs in target
raw_roads = raw_roads[!is.na(raw_roads[[y_name]]),]

run_IVreg = function(raw_roads){
  linear_controls = c('left','right')
  controls = c('primary_school','med_center','elect','tdist',
               'irr_share','ln_land','pc01_lit_share',
               'pc01_sc_share','bpl_landed_share','bpl_inc_source_sub_share',
               'bpl_inc_250plus','vhg_dist_id')

  raw_roads$vhg_dist_id = as.factor(raw_roads$vhg_dist_id)
  iv_form = paste0(y_name,' ~ r2012 + ',paste0(controls,collapse = ' + '),' + ',paste0(linear_controls,collapse = ' + '),
                   ' | ', paste0(controls,collapse = ' + '),' + t + ',paste0(linear_controls,collapse = ' + '))

  ivmodel=ivreg(as.formula(iv_form), x=TRUE, data=raw_roads)
  clustered_ses = cluster.robust.se(ivmodel, raw_roads$vhg_dist_id)
  return(clustered_ses['r2012',])
}

all_results = list()
all_results[['All states']] = run_IVreg(raw_roads)
state_subsets = unique(raw_roads$pc01_state_name)

for (s in state_subsets){
  all_results[[str_to_title(s)]] = run_IVreg(raw_roads[pc01_state_name == s])
}
all_results = as.data.frame(t(as.data.frame(all_results)))
all_results$State = rownames(all_results)
fwrite(all_results,paste0(OUTDIR,'quartile_ivreg_results.csv'))