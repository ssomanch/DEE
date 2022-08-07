# Directories
seed1_root=../output/simulation_1/$(seed1)
all_params_root=$(seed1_root)/$(CATE_ls)/$(bias_ls)

step1_output=$(seed1_root)/LORD3_inputs.csv
step2a_output=$(seed1_root)/LORD3_results.csv
step2b_output=$(seed1_root)/voroni_KNN_centers__ignore_TE_estimates.csv
step3_output=$(all_params_root)/LORD3_inputs_and_CATE_bias_Y.csv
step4_output=$(all_params_root)/TE_bias_estimates_at_voroni_KNN_centers.csv
step4b_output=$(all_params_root)/unfiltered_estimates_in_L.csv

# Two versions of steps 5 and 6 -- using filtered and unfiltered estimates
step5_filtered_output=$(all_params_root)/bias_correction_description.csv
step6_filtered_output=$(all_params_root)/PLP_strategy_search.csv

step5_unfiltered_output=$(all_params_root)/bias_correction_description_unfiltered.csv
step6_unfiltered_output=$(all_params_root)/PLP_strategy_search_unfiltered.csv

# Cattaneo benchmark
cattaneo_benchmark_description=$(all_params_root)/cattaneo_benchmark_description.csv
# Kallus benchmark
kallus_benchmark_description=$(all_params_root)/kallus_benchmark_description.csv

.DEFAULT_GOAL=all

# Step 1: Sample X, G, and D
$(step1_output): src/step1_get_X_and_D_for_LORD3.py
	mkdir -p $(@D)
	python src/step1_get_X_and_D_for_LORD3.py --seed1=$(seed1)

# Step 2a: Apply LORD3
$(step2a_output): src/step2a_run_LORD3.R $(step1_output)
	mkdir -p $(@D)
	/usr/local/bin/Rscript src/step2a_run_LORD3.R $(seed1)

# Step 2b: Apply Alg. 1: Voroni KNN 
$(step2b_output): src/step2b_compute_voroni_KNN_centroids.R $(step2a_output)
	mkdir -p $(@D)
	/usr/local/bin/Rscript src/step2b_compute_voroni_KNN_centroids.R $(seed1)

# Step 3: Sample Y for training points + test grid + Voroni/KNN centers
$(step3_output): src/step3_sample_CATE_bias_and_Y.py $(step2b_output)
	mkdir -p $(@D)
	python src/step3_sample_CATE_bias_and_Y.py --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls)

# Step 4: Compute TE along RD, fit observational esitmator (CF) and compute bias
$(step4_output): src/step4_compute_TE_and_bias_at_voroni_centers.R $(step3_output)
	/usr/local/bin/Rscript src/step4_compute_TE_and_bias_at_voroni_centers.R $(seed1) $(CATE_ls) $(bias_ls)

# Step 4b: Compute TE at all points in L -- not just Voroni/KNN points in U -- and compute bias
$(step4b_output): src/step4b_compute_TE_and_bias_at_points_in_L.R $(step4_output)
	/usr/local/bin/Rscript src/step4b_compute_TE_and_bias_at_points_in_L.R $(seed1) $(CATE_ls) $(bias_ls)

# Step 5: Compute the final estimators
$(step5_filtered_output): src/step5_get_final_CATE_estimates.py $(step4_output)
	python src/step5_get_final_CATE_estimates.py --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls)

# Step 6: Compute the final weighted estimators
$(step6_filtered_output): src/step6_compare_BMA_strategies.py $(step4_output)
	python src/step6_compare_BMA_strategies.py --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls)

# Step 5: Compute the final estimators for the unfiltered sets
$(step5_unfiltered_output): src/step5_get_final_CATE_estimates.py $(step4b_output)
	python src/step5_get_final_CATE_estimates.py --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls) --unfiltered

# Step 6: Compute the final weighted estimators for the unfiltered sets
$(step6_unfiltered_output): src/step6_compare_BMA_strategies.py $(step4b_output)
	python src/step6_compare_BMA_strategies.py --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls) --unfiltered

# Make the Cattaneo benchmark description
$(cattaneo_benchmark_description): src/benchmark_against_Cattaneo.R $(step3_output)
	/usr/local/bin/Rscript src/benchmark_against_Cattaneo.R $(seed1) $(CATE_ls) $(bias_ls)

# Make the kallus benchmark description
$(kallus_benchmark_description): src/benchmark_against_Kallus.R $(step3_output)
	/usr/local/bin/Rscript src/benchmark_against_Kallus.R $(seed1) $(CATE_ls) $(bias_ls)

all: $(step5_filtered_output) $(step6_filtered_output)\
	 $(step5_unfiltered_output) $(step6_unfiltered_output)\
	 $(cattaneo_benchmark_description)\
	 $(kallus_benchmark_description)