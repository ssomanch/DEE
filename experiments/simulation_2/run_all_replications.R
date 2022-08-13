set.seed(2212020)
N_iter=2
seed1_set = sample(.Machine$integer.max,N_iter)

CATE_ls_grid = c('0.2','0.5')#3.0')
bias_ls_grid = c('0.2','0.5')#3.0')

for(seed1 in seed1_set){
  for(CATE_ls in CATE_ls_grid){
    for(bias_ls in bias_ls_grid){
      OUTDIR = paste0('output/simulation_2/',seed1,'/',CATE_ls,'/',bias_ls,'/')
      dir.create(OUTDIR, showWarnings = FALSE, recursive=TRUE)
      command = paste0('make -f experiments/simulation_2/run_one_replication.makefile ',
                        ' seed1=',seed1,
                        ' CATE_ls=',CATE_ls,
                        ' bias_ls=',bias_ls)
      print(command)
      system(command,wait=T)
    }
  }
}
