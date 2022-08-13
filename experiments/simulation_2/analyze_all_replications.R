library(data.table)
library(stringr)
library(tidyr)
library(ggplot2)
library(dplyr)
library(latex2exp)

EXPERIMENT = 'simulation_2'
ix_cols = c('seed1','CATE_ls','bias_ls','fname')

valid_LSs = c('0.2','0.5')
y_upper = c(.9,0.9,0.65,0.5)

OUTDIR = paste0('../output/',EXPERIMENT,'/summary_figures/')
dir.create(OUTDIR, showWarnings = FALSE)

################################################################
# Utility functions from stack for faceting gtable             #
################################################################

library(grid)
library(gridExtra)
library(gtable)
library(ggplot2)

GeomTable <- ggproto(
  "GeomTable",
  Geom,
  required_aes = c("x", "y",  "table"),
  default_aes = aes(
    widthx = 10,
    widthy = 10,
    basesize = 8,
    rownames = NA
  ),
  draw_key = draw_key_blank,

  draw_panel = function(data, panel_scales, coord) {
    if (nrow(data) != 1) {
      stop(
        sprintf(
          "only one table per panel allowed, got %s (%s)",
          nrow(data),
          as.character(data)
        ),
        call. = FALSE
      )
    }
    wy = data$widthy / 2
    wx = data$widthx / 2

    corners <-
      data.frame(x = c(data$x - wx, data$x + wx),
                 y = c(data$y - wy, data$y + wy))
    d <- coord$transform(corners, panel_scales)
    # gross hack, but I've found no other way to get a
    # table/matrix/dataframe to this point :-(
    table = read.csv(text = data$table, header = TRUE)
    if (!is.na(data$rownames)) {
      rownames(table) <-
        unlist(strsplit(data$rownames, "|", fixed = TRUE))
    }

    x_rng <- range(d$x, na.rm = TRUE)
    y_rng <- range(d$y, na.rm = TRUE)

    vp <-
      viewport(
        x = mean(x_rng),
        y = mean(y_rng),
        width = diff(x_rng),
        height = diff(y_rng),
        just = c("center", "center")
      )

    grob <-
      tableGrob(table, theme = ttheme_minimal(base_size = data$basesize))
    # add a line across the header
    grob <- gtable_add_grob(
      grob,
      grobs = segmentsGrob(y1 = unit(0, "npc"),
                           gp = gpar(lwd = 2.0)),
      t = 1,
      b = 1,
      l = 1,
      r = ncol(table) + 1
    )
    editGrob(grob, vp = vp, name = paste(grob$name, facet_id()))
  }
)

facet_id <- local({
  i <- 1
  function() {
    i <<- i + 1
    i
  }
})

geom_table <-
  function(mapping = NULL,
           data = NULL,
           stat = "identity",
           position = "identity",
           na.rm = FALSE,
           show.legend = NA,
           inherit.aes = TRUE,
           ...) {
    layer(
      geom = GeomTable,
      mapping = mapping,
      data = data,
      stat = stat,
      position = position,
      show.legend = show.legend,
      inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, ...)
    )
  }


# helper function
to_csv_ <- function(x) {
    paste(capture.output(write.csv(x, stdout(), row.names = F)), 
          collapse = "\n")
  }

################################################################
# Utility functions for loading filtered/unfiltered outputs    #
################################################################


get_PLP_search_results = function(filtered){
	# Load the weighted results, for either the filtered or the unfilterd
	# runs. Adds a column labeled 'Filtered'
	if (filtered){
		all_outputs = list.files(paste0('../output/',EXPERIMENT),'PLP_strategy_search.csv',recursive = T)
	} else {
		all_outputs = list.files(paste0('../output/',EXPERIMENT),'PLP_strategy_search_unfiltered.csv',recursive = T)
	}
		
	PLP_search = rbindlist(lapply(str_split(all_outputs,'/'),function(x) as.data.frame(t(x))))
	colnames(PLP_search) = ix_cols
	PLP_search[,path:=all_outputs]
	
	all_descrs = list()
	for (i in 1:nrow(PLP_search)){
		descr = t(read.csv(paste0('../output/',EXPERIMENT,'/',PLP_search[i,path]),row.names = 1))
		df = as.data.frame(descr)
		cols = colnames(df)
		df$strategy = rownames(df)
		df = pivot_wider(df,names_from='strategy',values_from = cols)
		all_descrs[[i]] = df
	}
	
	PLP_search = cbind(PLP_search,rbindlist(all_descrs,fill=T))
	PLP_search[,Filtered:=filtered]
	return(PLP_search)
}


get_original_results = function(filtered){
	## Original results (unmixed baselines)
	if (filtered){
		all_outputs = list.files(paste0('../output/',EXPERIMENT),'*_description.csv',recursive = T)
	} else {
		all_outputs = list.files(paste0('../output/',EXPERIMENT),'*_description_unfiltered.csv',recursive = T)
	}
	
	seeds_configs_and_paths = rbindlist(lapply(str_split(all_outputs,'/'),function(x) as.data.frame(t(x))))
	colnames(seeds_configs_and_paths) = ix_cols
	seeds_configs_and_paths$path = all_outputs
	
	all_descrs = list()
	for (i in 1:nrow(seeds_configs_and_paths)){
		descr = t(read.csv(paste0('../output/',EXPERIMENT,'/',seeds_configs_and_paths[i,path]),row.names = 1))
		all_descrs[[i]] = as.data.frame(descr)
	}
	all_descrs = rbindlist(all_descrs,fill=T)
	seeds_configs_and_paths = cbind(seeds_configs_and_paths,all_descrs)
	
	if (filtered){
		seeds_configs_and_paths[,method:=plyr::mapvalues(fname,
														 from=c('bias_correction_description.csv',
															    'CATE_extrapolation_description.csv',
															    'cf_ignoring_RDs_description.csv',
															    'kallus_benchmark_description.csv'),
														 to=c('Bias','CATE','Causal forest','Kallus'))]
	} else {
		seeds_configs_and_paths[,method:=plyr::mapvalues(fname,
														from=c('bias_correction_description_unfiltered.csv',
															'CATE_extrapolation_description_unfiltered.csv',
															'cf_ignoring_RDs_description_unfiltered.csv'),
														to=c('Bias','CATE','Causal forest'))]
	}
	
	# Compute the mean MSE using causal forest directly
	CF_means = seeds_configs_and_paths %>%
				filter(method=='Causal forest') %>%
				group_by(CATE_ls,bias_ls) %>%
				summarise(m = mean(MSE),se=sd(MSE)/sqrt(n()),n=n())
	CF_means$Strategy = 'BW'

	# Compute the mean MSE using Kallus
	Kallus_means = seeds_configs_and_paths %>%
				filter(method=='Kallus') %>%
				group_by(CATE_ls,bias_ls) %>%
				summarise(m = mean(MSE),se=sd(MSE)/sqrt(n()),n=n())
	Kallus_means$Strategy = 'BW'
	
	seeds_configs_and_paths = seeds_configs_and_paths %>%
		filter(method %in% c('CATE','Bias')) %>%
		select(c(ix_cols,'MSE','method')) %>%
		mutate(Mixture='Baseline',Strategy = method,value=MSE)
	
	
	seeds_configs_and_paths = seeds_configs_and_paths[,Filtered:=filtered]
	seeds_configs_and_paths = seeds_configs_and_paths[(CATE_ls %in% valid_LSs) & (bias_ls %in% valid_LSs)]

	return(list(seeds_configs_and_paths=seeds_configs_and_paths,
				CF_means=CF_means,
				Kallus_means=Kallus_means))
}

################################################################
# First set of figures: weights on bias model using one of     #
# four model weighting strategies: either MLL, LOO, 1-MC       #
# buffering, or BW                                             #
################################################################

filtered_PLP_search = get_PLP_search_results(T)
unfiltered_PLP_search = get_PLP_search_results(F)
filtered_PLP_search = filtered_PLP_search[(CATE_ls %in% valid_LSs) & (bias_ls %in% valid_LSs)]
unfiltered_PLP_search = unfiltered_PLP_search[(CATE_ls %in% valid_LSs) & (bias_ls %in% valid_LSs)]
both_PLP_search = rbind(filtered_PLP_search,unfiltered_PLP_search,fill=T)

p = ggplot(filtered_PLP_search %>% mutate(`CATE lengthscale`=CATE_ls,`Bias lengthscale` = bias_ls),
		   aes(x=MLL_Percent.bias.in.mixture)) + geom_histogram(breaks=seq(0,1,0.1)) +
	facet_wrap(`CATE lengthscale` ~ `Bias lengthscale`,labeller = 'label_both') + theme(axis.text.x = element_text(angle = 90)) +
	xlab(TeX('Weight on bias model in mixture')) + ylab('Count') + ylim(0,50) + 
	ggtitle('Model averaging using marginal likelihood weights')
ggsave(paste0(OUTDIR,'MLL_model_bias_weights.png'),p,width=7,height=4)

p = ggplot(filtered_PLP_search %>% mutate(`CATE lengthscale`=CATE_ls,`Bias lengthscale` = bias_ls),
		   aes(x=LOO_Percent.bias.in.mixture)) + geom_histogram(breaks=seq(0,1,0.1)) +
	facet_wrap(`CATE lengthscale` ~ `Bias lengthscale`,labeller = 'label_both') + theme(axis.text.x = element_text(angle = 90)) +
	xlab('Weight on bias model in mixture') + ylab('Count') + ylim(0,50) + 
	ggtitle('Model averaging using LOOCV likelihood weights')
ggsave(paste0(OUTDIR,'LOO_model_bias_weights.png'),p,width=7,height=4)

p = ggplot(filtered_PLP_search %>% mutate(`CATE lengthscale`=CATE_ls,`Bias lengthscale` = bias_ls),
		   aes(x=`min dist_Percent.bias.in.mixture`)) + geom_histogram(breaks=seq(0,1,0.1)) +
	facet_wrap(`CATE lengthscale` ~ `Bias lengthscale`,labeller = 'label_both') + theme(axis.text.x = element_text(angle = 90)) +
	xlab('Weight on bias model in mixture') + ylab('Count') + ylim(0,50) + 
	ggtitle('Model averaging using weighted BLOOCV likelihood weights')
ggsave(paste0(OUTDIR,'dist_weighted_model_bias_weights.png'),p,width=7,height=4)

p = ggplot(filtered_PLP_search %>% mutate(`CATE lengthscale`=CATE_ls,`Bias lengthscale` = bias_ls),
		   aes(x=`random dist_Percent.bias.in.mixture`)) + geom_histogram(breaks=seq(0,1,0.1)) +
	facet_wrap(`CATE lengthscale` ~ `Bias lengthscale`,labeller = 'label_both') + theme(axis.text.x = element_text(angle = 90)) +
	xlab('Weight on bias model in mixture') + ylab('Count') + ylim(0,50) + 
	ggtitle('Model averaging using 1-MC BLOOCV likelihood weights')
ggsave(paste0(OUTDIR,'1MC_b_PPL_model_bias_weights.png'),p,width=7,height=4)

# New version that shows off-diagonal difference in log likelihoods

weighters = c('min dist','random dist','LOO','MLL')

for (weighter in weighters){
	delta = filtered_PLP_search[[paste0(weighter,'_CATE')]] - filtered_PLP_search[[paste0(weighter,'_bias')]]
	filtered_PLP_search[[paste0(weighter,'_delta')]] = delta
}

to_melt = filtered_PLP_search[,c('CATE_ls','bias_ls','min dist_delta','random dist_delta','LOO_delta','MLL_delta'),with=F]
melted = melt(to_melt,id.vars=c('CATE_ls','bias_ls'))
to_plot = melted %>% group_by(CATE_ls,bias_ls,variable) %>% summarize(mean_se(value,mult = 1.96))
to_plot[['Likelihood']] = plyr::mapvalues(to_plot$variable,
																	 c('min dist_delta','random dist_delta','LOO_delta','MLL_delta'),
																	 c('BW','1-MC','LOO','MLL'))
to_plot[['Bias lengthscale']] = to_plot$bias_ls
to_plot[['CATE lengthscale']] = to_plot$CATE_ls

to_plot = to_plot[(to_plot$bias_ls != to_plot$CATE_ls),]

 p = ggplot(to_plot,aes(x=Likelihood,y=y,ymin=ymin,ymax=ymax,color=Likelihood)) + 
		geom_errorbar() + facet_wrap(`CATE lengthscale`~`Bias lengthscale`,scales='free',labeller = label_both) + geom_point()+
		ylab('(CATE log likelihood) - (bias log likelihood)\n Mean and 95% CI')

ggsave(paste0(OUTDIR,'likelihood_deltas.png'),p,width=7,height=3)


# Relative MSE reduction
p = ggplot(filtered_PLP_search %>%
			filter(CATE_ls == bias_ls) %>%
			mutate(`CATE lengthscale`=CATE_ls,`Bias lengthscale` = bias_ls) %>%
			mutate(relative_MSE = `MLL_Mixed.MSE`/`MLL_Zero.One.MSE`) %>%
			mutate(relative_MSE_smooth = ifelse(`CATE lengthscale` == `Bias lengthscale`,
																					relative_MSE, NA)),
			aes(x=`min dist_Percent.bias.in.mixture`,
					y=relative_MSE)) + geom_smooth(aes(y = relative_MSE_smooth),span=2.5) + 
					geom_point(aes(y=relative_MSE),alpha=0.2) +
	facet_wrap(`CATE lengthscale` ~ `Bias lengthscale`,
			   nrow=1,labeller = 'label_both', scales='free_y') +
	theme(axis.text.x = element_text(angle = 90)) +
	xlab('Weight on bias model in mixture') + ylab('(Mixture MSE)/(Max likelihood MSE)') + 
	ggtitle('Model weights vs. MSE using MLL model averaging') +
	geom_hline(aes(yintercept = 1), linetype="dashed", color="darkgrey", size=0.5)
ggsave(paste0(OUTDIR,'relative_MSE_reduction_mixture.png'),p,width=7,height=3)


################################################################
# Second figure: mean and SE MSEs over the test grid           #
################################################################

original_filtered = get_original_results(T)
original_unfiltered = get_original_results(F)
	
seeds_configs_and_paths = rbind(original_filtered$seeds_configs_and_paths,
								original_unfiltered$seeds_configs_and_paths)
CF_means = original_filtered$CF_means
Kallus_means = original_filtered$Kallus_means

## Melt and stack to generate MSE boxplot figure
CF_means = original_filtered$CF_means

MSE = both_PLP_search[,c(ix_cols,'Filtered',colnames(both_PLP_search)[grepl('MSE',colnames(both_PLP_search))]),with=F]
MSE = pivot_longer(MSE,cols = colnames(MSE)[grepl('MSE',colnames(MSE))])

strat_and_mix = str_split(MSE$name,'_',simplify=T)
colnames(strat_and_mix) = c('Strategy','Mixture')
MSE = cbind(MSE,strat_and_mix) %>% mutate(Mixture = ifelse(grepl('Mixed',Mixture),'Mixture','Zero one'))

to_plot = rbindlist(list(MSE,seeds_configs_and_paths),fill=T) %>%
		   mutate(`CATE lengthscale`=CATE_ls,`Bias lengthscale` = bias_ls) %>%
		   filter(Mixture %in% c('Baseline','Mixture')) %>%
		   filter(Strategy %in% c('Bias','CATE','MLL','LOO','random dist','min dist')) %>%
		   mutate(Strategy = ifelse(Strategy == 'random dist','1-MC',
		   							ifelse(Strategy == 'min dist','BW',Strategy))) %>%
		   mutate(Strategy = factor(Strategy,levels=c("Bias",'CATE','MLL','LOO','1-MC','BW')))

both_sum_stats = rbind(CF_means %>% mutate(Benchmark='CF'),
					   Kallus_means %>% mutate(Benchmark='Kallus'))

# Make tables for annotation
cate_v = rep(NA,length(valid_LSs)^2)
bias_v = rep(NA,length(valid_LSs)^2)
tables = rep(NA,length(valid_LSs)^2)
i=1
for (cls in valid_LSs){
	for (bls in valid_LSs){
		subdf = both_sum_stats %>% 
							filter((CATE_ls == cls) & (bias_ls == bls)) %>% 
							mutate(`MSE (SE)`=paste0(round(m,2), ' (',round(se,2),')'))
		tb = t(subdf[,c('MSE (SE)')])
		
		colnames(tb) = t(subdf[,c('Benchmark')])
		
		tables[i] = to_csv_(tb)
		bias_v[i] = bls
		cate_v[i] = cls
		i = i+1
	}
}

annot_t = data.frame(bias_ls = as.character(bias_v),
					 CATE_ls = as.character(cate_v),
					 Filtered = F,
					 t = tables,
					 stringsAsFactors = FALSE)
# Set up ylims via blank data
blank_data = tibble(`Bias lengthscale` = bias_v,
					`CATE lengthscale` = cate_v,
					Filtered = F,
					x = 1, y = y_upper)
blank_data0 = tibble(`Bias lengthscale` = bias_v,
					 `CATE lengthscale` = cate_v,
					 Filtered = F,
					 x = 1, y = 0)

p = ggplot(to_plot,aes(y=value,x=Strategy,color=Filtered)) + 
	stat_summary(fun.data = mean_se, geom = "errorbar", fun.args=list(mult=1.96)) +
	stat_summary(fun.data = mean_se, geom = "point", fun.args=list(mult=1.96))+
	facet_wrap(`CATE lengthscale` ~ `Bias lengthscale`,labeller = 'label_both',scales='free_y') + 
	theme(axis.text.x = element_text(angle = 90,hjust = 0.95,vjust=0.5)) +
	xlab('Model selection strategy') + 
	ylab('MSE mean and 95% CI') + geom_blank(data = blank_data, aes(x = x, y = y)) + 
	geom_blank(data = blank_data0, aes(x = x, y = y)) + 
	ggtitle('Comparing MSEs for different model selection strategies') +
	geom_table(data = annot_t %>%
			            mutate(`Bias lengthscale`=bias_ls,
							   `CATE lengthscale`=CATE_ls),
			   aes(table = t), x = 3.5, y = 0.875*y_upper, rownames = "MSE (SE)",basesize=6) +
	guides(color=guide_legend(title="Alg. 1 applied"))

ggsave(paste0(OUTDIR,'model_selection_MSE_comparison.png'),p,width=7,height=4)
