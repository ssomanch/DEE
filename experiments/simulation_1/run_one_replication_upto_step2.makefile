# Directories
seed1_root=output/simulation_1/$(seed1)
all_params_root=$(seed1_root)/$(CATE_ls)/$(bias_ls)

step1_output=$(seed1_root)/LORD3_inputs.csv
step2a_output=$(seed1_root)/LORD3_results.csv
step2b_output=$(seed1_root)/voroni_KNN_centers__ignore_TE_estimates.csv


.DEFAULT_GOAL=all

# Step 1: Sample X, G, and D
$(step1_output): experiments/simulation_1/step1_get_X_and_D_for_LORD3.py
	mkdir -p $(@D)
	python experiments/simulation_1/step1_get_X_and_D_for_LORD3.py --seed1=$(seed1) --sample-at-RDD 1

# Step 2a: Apply LORD3
$(step2a_output): experiments/simulation_1/step2a_run_LORD3.R $(step1_output)
	mkdir -p $(@D)
	Rscript experiments/simulation_1/step2a_run_LORD3.R $(seed1)

# Step 2b: Apply Alg. 1: Voroni KNN 
$(step2b_output): experiments/simulation_1/step2b_compute_voroni_KNN_centroids.R $(step2a_output)
	mkdir -p $(@D)
	Rscript experiments/simulation_1/step2b_compute_voroni_KNN_centroids.R $(seed1)

all: $(step2b_output)