.PHONY: setup_renv setup_python_venv simulation_1 simulation_2 rural_roads

# Step 1: Setup the R environment
setup_renv:
	mkdir -p renv/cellar
	R CMD build LORD3 
	mv LORD3_0.1.0.tar.gz renv/cellar/
	R -e "renv::restore()"
	
setup_python_venv:
	python3 -m venv DEE_env
	source DEE_env/bin/activate && pip install -r requirements.txt
	
# Step 2: Create test grid for benchmarking
output/test_grid.csv: utils/get_test_x.py
	mkdir output/
	python utils/get_test_x.py
	
# Step 2: Run simulation 1
simulation_1: output/test_grid.csv
	Rscript experiments/simulation_1/run_all_replications.R
	Rscript experiments/simulation_1/analyze_all_replications.R
		
# Step 3: Run simulation 2
simulation_2: output/test_grid.csv
	Rscript experiments/simulation_2/run_all_replications.R
	Rscript experiments/simulation_2/analyze_all_replications.R

all: simulation_1 simulation_2