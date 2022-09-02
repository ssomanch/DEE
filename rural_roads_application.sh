#!/bin/bash

ys=( transport_index_andrsn occupation_index_andrsn agriculture_index_andrsn consumption_index_andrsn firms_index_andrsn )
RD_types=( population space_and_population )
estimators=( nonparametric_estimator rotated_2SLS )
k_primes=( 400 )
t_partition=( 40 )

for y in "${ys[@]}"
do
  for RD_type in "${RD_types[@]}" 
  do
    for estimator in "${estimators[@]}"
    do
      for k in "${k_primes[@]}"
      do
        for t in "${t_partition[@]}"
        do
          make -f rural_roads_makefile index=$y RD_type=$RD_type estimator=$estimator k_prime=$k t_partition=$t
        done
      done
    done
  done
done
