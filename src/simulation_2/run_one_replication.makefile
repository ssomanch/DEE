# Directories
all_params_root=../output/simulation_2/$(seed1)/$(CATE_ls)/$(bias_ls)

step1_output=$(all_params_root)/LORD3_inputs.csv
step2a_output=$(all_params_root)/LORD3_results.csv
step2b_output=$(all_params_root)/TE_estimates_at_voroni_KNN_centers_without_true_parameters.csv
step3_output=$(all_params_root)/true_CATE_and_bias_at_voroni_KNN_centers.csv
step4_output=$(all_params_root)/TE_bias_estimates_at_voroni_KNN_centers_with_true_parameters.csv
step4b_output=$(all_params_root)/unfiltered_estimates_in_L.csv

# Two versions of steps 5 and 6 -- using filtered and unfiltered estimates
step5_filtered_output=$(all_params_root)/bias_correction_description.csv
step6_filtered_output=$(all_params_root)/PLP_strategy_search.csv

step5_unfiltered_output=$(all_params_root)/bias_correction_description_unfiltered.csv
step6_unfiltered_output=$(all_params_root)/PLP_strategy_search_unfiltered.csv

# Kallus benchmark
kallus_benchmark_description=$(all_params_root)/kallus_benchmark_description.csv

.DEFAULT_GOAL=all

# Step 1: Sample functions and data
$(step1_output): src/step1_sample_dataset_CATE_and_bias.py
	python -m src.step1_sample_dataset_CATE_and_bias --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls)

# Step 2a: Run LORD3
$(step2a_output): src/step2a_run_LORD3.R $(step1_output)
	/usr/local/bin/Rscript src/step2a_run_LORD3.R $(seed1) $(CATE_ls) $(bias_ls)

# Step 2b: Run Voroni KNN. Note for this simulation this yields the actual CATE estimates,
# since in this DGP treatment and potential outcomes are dependent, and thus are both sampled
# in step 1 (e.g. D varies across the four simulations).
$(step2b_output): src/step2b_compute_voroni_KNN_centroids_and_TE_estimate.R $(step2a_output)
	/usr/local/bin/Rscript src/step2b_compute_voroni_KNN_centroids_and_TE_estimate.R $(seed1) $(CATE_ls) $(bias_ls)

# Step 3: Compute true bias and CATE for Voroni/KNN centers
$(step3_output): src/step3_sample_CATE_bias_and_Y.py $(step2b_output)
	python -m src.step3_sample_CATE_bias_and_Y --seed1=$(seed1) --CATE-ls=$(CATE_ls) --bias-ls=$(bias_ls)

# Step 4: Fit observational esitmator (CF), estimate bias at Voroni/KNN centers
$(step4_output): src/step4_fit_CF_and_estimate_bias_at_voroni_centers.R $(step3_output)
	/usr/local/bin/Rscript src/step4_fit_CF_and_estimate_bias_at_voroni_centers.R $(seed1) $(CATE_ls) $(bias_ls)

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

# Make the kallus benchmark description
$(kallus_benchmark_description): src/benchmark_against_Kallus.R $(step1_output) $(step2a_output)
	/usr/local/bin/Rscript src/benchmark_against_Kallus.R $(seed1) $(CATE_ls) $(bias_ls)

all: $(step5_filtered_output) $(step6_filtered_output)\
	 $(step5_unfiltered_output) $(step6_unfiltered_output)\
	 $(kallus_benchmark_description)