library(data.table)
library(lsa)

COS_POP_THRESH = 0.75


est_set = fread('output/rural_roads/roads.csv')
LORD3_results = fread('output/rural_roads/roads_LORD3.csv')

bound = data.frame(cbind(est_set,LORD3_results))

nvec = bound[,c("longitude_nvec","latitude_nvec","total_pop_nvec")]
nvec = as.matrix(nvec)
nvec = nvec / sqrt(rowSums(nvec^2))
target_v = c(0,0,1)

abs_cos_sim = abs(apply(nvec,1,function(r) cosine(r,target_v)))

pop_RDD = abs_cos_sim > COS_POP_THRESH
                        
LORD3_results$total_pop_RD = pop_RDD
                        
fwrite(LORD3_results,'output/rural_roads/roads_LORD3_w_RDD_category.csv')