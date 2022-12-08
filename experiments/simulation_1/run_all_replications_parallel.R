library(doParallel)

set.seed(2212020)
N_iter=50
n_cores = 20
seed1_set = sample(.Machine$integer.max,N_iter)

CATE_ls_grid = c('0.2', '0.5')
bias_ls_grid = c('0.2', '0.5')

# Create all the directories first 

for(seed1 in seed1_set){
  for(CATE_ls in CATE_ls_grid){
    for(bias_ls in bias_ls_grid){
      OUTDIR = paste0('output/simulation_1/',seed1,'/',CATE_ls,'/',bias_ls,'/')
      dir.create(OUTDIR, showWarnings = FALSE, recursive=TRUE)
    }
  }
}

# Run simulation 1 for a set of CATE and bias length scales
run_simulation <- function(seed1) {
  for(CATE_ls in CATE_ls_grid){
    for(bias_ls in bias_ls_grid){
      command = paste0('make -f experiments/simulation_1/run_one_replication.makefile ',
                        ' seed1=',seed1,
                        ' CATE_ls=',CATE_ls,
                        ' bias_ls=',bias_ls)
      print(command)
      system(command,wait=T)
    }
  }
}

registerDoParallel(cores=n_cores)

# Run the simulation for all the seeds in parallel

nothing <- foreach(seed1=seed1_set) %dopar% {
  run_simulation(seed1)
}