library(data.table)
library(FNN)
library(LORD3)
library(dplyr)
library(tidyr)
library(Matrix)
library(AER)
library(pracma)
library(gridExtra)
library(ggplot2)
library(ggforce)


source('../neighborhood_and_index_set_selection_utils.R')

options(dplyr.summarise.inform = FALSE)
seed1 = 1009807543
CATE_ls = '0.2'
bias_ls = '0.2'
M_prime = 400
k_prime = 1000
t_partition = 30
estimator = 'rotated_2SLS'
nvec_col_names = c('raw_nvec1','raw_nvec2')

est_set = fread(paste0('../output/simulation_1/',seed1,'/LORD3_inputs.csv'))
LORD3_results = fread(paste0('../output/simulation_1/',seed1,'/LORD3_results.csv'))

df = cbind(est_set,LORD3_results)
df = df[order(-LLR)]

L_x = as.matrix(df[1:M_prime,.(X1,X2)])
L_v = as.matrix(df[1:M_prime,nvec_col_names,with=F])

dir.create('../output/simulation_1/summary_figures/voroni_evolution/', showWarnings = FALSE)
voroni_assets = get_voroni_knn_discontinuities_index_sets_and_estimates(
	L_x, L_v, df, k_prime, t_partition, estimator,
	save_plots='../output/simulation_1/summary_figures/voroni_evolution/',
	R2_figure=T
)

p1 = ggplot(df[1:M_prime],aes(x=X1,y=X2)) +
			geom_point() + coord_equal()  + xlim(c(0,1)) + ylim(c(0,1)) + 
			ggtitle('Step 1: M=400 discovered discontinuities')

voroni_plot = plot_voroni_from_Xs_and_neighbors_and_sides(
	df[,.(X1,X2)],
	voroni_assets$neighbors_and_sides
)
p2 = voroni_plot + geom_point(data=voroni_assets$top_M_prime_TEs,aes(x=X1,y=X2),inherit.aes = F) +
			coord_equal() + ggtitle('Step 2: Alg. 1 output with index sets') + xlim(c(0,1)) + ylim(c(0,1))

m = grid.arrange(p1,p2,widths=c(4,4),heights=c(4))
ggsave('../output/simulation_1/summary_figures/grid_step1_step2.png',m,width=10,height=5)


# New figure per Daniels suggestions
size_U = nrow(voroni_assets$top_M_prime_TEs)
neighs = get.knn(df[,.(X1,X2)],200)
ball_rs = apply(neighs$nn.dist,1,max)[1:size_U]
in_a_ball = sort(unique(c(1:size_U,as.vector(neighs$nn.index[1:size_U,]))))
freq_dt = df[in_a_ball,]
freq_dt[,Center:=(1:nrow(freq_dt))<=size_U]
freq_dt[,R:=c(ball_rs,rep(NA,nrow(freq_dt) - length(ball_rs)))]

freq_dt[,Side:=0]
freq_dt[,freq:=0]
for (i in 1:size_U){
	cx = unlist(freq_dt[i,.(X1,X2)])
	nv = unlist(freq_dt[i,.(raw_nvec1,raw_nvec2)])
	r = ball_rs[i]
	cXs = as.matrix(sweep(freq_dt[,.(X1,X2)],2,cx))
	in_ball = sqrt(rowSums(cXs^2)) <= r
	print(sum(in_ball))
	side = bisect_neighborhood(cXs[in_ball,],nv)
	side = as.numeric(reorder_side(cXs[in_ball,],side))
  freq_dt[in_ball,Side:=Side + side]
	freq_dt[in_ball,freq:=freq + 1]
}

freq_dt[,Side:=(Side/freq >= 0.5)]

p1a = voroni_plot + geom_point(data=voroni_assets$top_M_prime_TEs,aes(x=X1,y=X2),inherit.aes = F) +
			coord_equal(xlim=c(0,1),ylim = c(0,1)) + ggtitle('Step 2: Alg. 1 output with index sets')

p2a = ggplot() + geom_point(data=freq_dt[Center==F,],aes(x=X1,y=X2,alpha=(Side+0.5)/2),color='cyan4') + 
			geom_point(data=freq_dt[Center==T,],aes(x=X1,y=X2),color='black') + 
			theme(legend.position = "none") + coord_equal(xlim=c(0,1),ylim = c(0,1)) + ggtitle("Top M' neighborhoods with KNNs")

for (i in 1:size_U){
	p2a = p2a + geom_circle(data=freq_dt[i,],aes(x0=X1,y0=X2,r=R),inherit.aes=F,size=0.25)
}

m = grid.arrange(p1a, p2a,widths=c(4,4),heights=c(4))
ggsave('../output/simulation_1/summary_figures/side_by_side_spread_and_overlap.png',m,width=10,height=5)


m = grid.arrange(p1, p2a,widths=c(4,4),heights=c(4))
ggsave('../output/simulation_1/summary_figures/L_to_reduced_L_with_overlap.png',m,width=10,height=5)

# New version with all three side by side

new_M_prime = nrow(voroni_assets$top_M_prime_TEs)
p2a = p2a + ggtitle(paste0('Limit to top M=',new_M_prime,' discontinuities'))
p2 = p2 + ggtitle(paste0('Step 2: VKNN returns ',new_M_prime,' discontinuities')) 

m = grid.arrange(p1, widths=c(4),heights=c(4))
ggsave('../output/simulation_1/summary_figures/just_left_1.png',m,width=4,height=4)

m = grid.arrange(p1, p2a, widths=c(4,4),heights=c(4))
ggsave('../output/simulation_1/summary_figures/just_left_2.png',m,width=8,height=4)

m = grid.arrange(p1, p2a, p2, widths=c(4,4,4),heights=c(4))
ggsave('../output/simulation_1/summary_figures/all_3_versions.png',m,width=12,height=4)