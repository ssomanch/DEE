library(ggplot2)
library(data.table)
library(lsa)

target_v = c(0,0,1)
# M_prime is chosen based on randomization testing by running 
# randomization_testing_LORD3.R and  compute_M_prime_from_randomization.R
# Note that these files are not run in the regular flow of the application.
M_prime = 2784
cos_pop_thresh = 0.75

est_set = fread('output/rural_roads/roads.csv')
old_LORD3_results = fread('output/rural_roads/roads_LORD3.csv')

bound = data.frame(cbind(est_set,old_LORD3_results))
bound = bound[order(-bound$LLR),]


top_M_nvec = bound[1:M_prime,c("longitude_nvec","latitude_nvec","total_pop_nvec")]
top_M_nvec = as.matrix(top_M_nvec)
top_M_nvec = top_M_nvec / sqrt(rowSums(top_M_nvec^2))

abs_cos_sim = abs(apply(top_M_nvec,1,function(r) cosine(r,target_v)))
p <- qplot(abs_cos_sim) + ylab('Number of discovered local RDs') +
  geom_vline(xintercept = cos_pop_thresh,linetype=2,color='red') + 
  xlab('|Cos. similarity| between RD normal vector and population std. basis vector') +
  ggtitle('Categorizing discovered RD based on RD normal vector')

ggsave('output/rural_roads/figures/cos_sim_plot.png',p, width=7,height=4)
                   
nvecs = bound[,c("longitude_nvec","latitude_nvec","total_pop_nvec")]
nvecs = as.matrix(nvecs)
nvecs = nvecs / sqrt(rowSums(nvecs^2))
abs_cos_sim = abs(apply(nvecs,1,function(r) cosine(r,target_v)))
pop_RDD = abs_cos_sim > cos_pop_thresh
bound[['RD type']] = ifelse(pop_RDD,'Population','Spatial')

hline = bound[M_prime,'LLR']

p <- ggplot(bound[bound$LLR>0,],aes(x=total_pop,y=LLR,color=`RD type`)) + geom_point(size=1,alpha=0.5) + 
  xlab('Village population') + ylab('LLR') + geom_hline(yintercept=hline,linetype=2,color='red') + 
  ggtitle('LORD3 LLR statistic by village population') 

ggsave('output/rural_roads/figures/LLR_plot.png',p, width=7,height=4)
                        
top_M = bound[1:M_prime,]
fwrite(top_M,'output/rural_roads/figures/top_M_for_map.csv')