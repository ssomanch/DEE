# README

## Introduction

This repo contains code for the following paper and replicating the experiments in the DEE paper. We will update the repo soon such that the method is available as a standalone package. 

B. Jakubowski, S. Somanchi, E. McFowland III, and D. B. Neill. Exploiting Discovered Regression Discontinuities to Debias Conditioned-on-Observable Estimators. <em>Journal of Machine Learning Research (JMLR)</em>, 24(133), pp.1-57, 2023.

## Instructions

### Building environment

This projects uses a `python venv` and `R renv` to manage dependencies. To build
the `R` and `python` environments, run

```
make setup_renv
make setup_python_venv
```

### Running simulations 1 and 2

To run simulations 1 and 2 in parallel, run

```
make simulation_1_parallel simulation_2_parallel
```
### Running Rural Roads Application

Here are the steps to run DEE over the Rural Roads [1] data:

#### Seeting up input data

1. Download the replication data files pmgsy_working_aer.dta and pmgsy_working_aer_main_sample.dta from the ICPSR data folder [https://www.openicpsr.org/openicpsr/project/109703/version/V2/view?path=/openicpsr/109703/fcr:versions/V2.1/data/pmgsy&type=folder] into the subdirectory
`DEE/data/rural_roads/`
2. Download the (Indian States shapefile from ArcGIS Hub)[https://hub.arcgis.com/datasets/ba24c0b6ade04b43aa1ca8dee504ee7e/explore?location=13.745128%2C73.382768%2C5.77] into the subdirectory `DEE/data/rural_roads/`. Rename the all the shape files as India_State_Boundary.*

The final directory structure should be

```
DEE/
└── data/rural_roads/
		├── pmgsy_working_aer.dta
		├── pmgsy_working_aer_mainsample.dta
		└── India_States_AMD1_GADM-shp/
			├── India_State_Boundary.shx
			├── India_State_Boundary.shp
			├── India_State_Boundary.prj
			├── India_State_Boundary.dbf
			└── India_State_Boundary.cpg
```
#### Running Rural Raods

To run rural roads, run

```
make rural_roads
```

To produce figures and summary statistics, run 

```
python experiments/rural_roads/make_main_py_figures.py
Rscript experiments/rural_roads/make_main_R_figures.R
python experiments/rural_roads/stack_neighborhood_descriptives.py
```

#### Acknowledgements

1. We ackowledge William Herlands for providing the code for LORD3 method [2] used extensively in our research. 


#### References

[1] Asher, Sam, and Paul Novosad. “Rural Roads and Local Economic Development.” American Economic Review 110, no. 3 (March 2020): 797–823. https://doi.org/10.1257/aer.20180268.

[2] Herlands, William, Edward McFowland III, Andrew Gordon Wilson, and Daniel B. Neill. "Automated local regression discontinuity design discovery." In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1512-1520. 2018.
