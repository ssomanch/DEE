library(data.table)
library(tidyr)

alpha = 0.05

LORD3_results = fread('output/rural_roads/roads_LORD3.csv')

randomization_max_LLRs = fread('output/rural_roads/roads_LORD3_randomization_max_LLRs.csv')

max_LLR_1_minus_alpha = quantile(randomization_max_LLRs$x, 1-alpha)

M_prime = sum(LORD3_results$LLR > max_LLR_1_minus_alpha)

print(paste0("The M prime value from randomization testing is ", M_prime))