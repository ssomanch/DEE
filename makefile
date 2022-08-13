.PHONY: install_LORD3 simulation_1 simulation_2 rural_roads

# Step 1: Build and install LORD3 R package
install_LORD3:
	R CMD build LORD3
	R CMD install LORD3_0.1.0.tar.gz
	
# Step 2: Run simulation 1
simulation_1:
	Rscript experiments/simulation_1/run_all_replications.R
	Rscript experiments/simulation_1/analyze_all_replications.R
	
		
# Step 3: Run simulation 2
simulation_2:
	Rscript experiments/simulation_2/run_all_replications.R
	Rscript experiments/simulation_2/analyze_all_replications.R

