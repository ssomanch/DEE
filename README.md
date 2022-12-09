# README

## Introduction

This repo contains code for DEE and the experiments in the DEE paper.

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
2. Download the (Indian States shapefile from ArcGIS Hub)[https://hub.arcgis.com/datasets/ba24c0b6ade04b43aa1ca8dee504ee7e/explore?location=13.745128%2C73.382768%2C5.77] or other Github repos like this[https://github.com/AnujTiwari/India-State-and-Country-Shapefile-Updated-Jan-2020] into the subdirectory `DEE/data/rural_roads/`. Rename the all the shape files as India_State_Boundary. 

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

#### References

[1] Asher, Sam, and Paul Novosad. “Rural Roads and Local Economic Development.” American Economic Review 110, no. 3 (March 2020): 797–823. https://doi.org/10.1257/aer.20180268.