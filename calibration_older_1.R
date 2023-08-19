library(ems)
library(dplyr)
library(forestplot)

db_loop_list <- list("MIMIC_eICU", "eICU", "Ams", "MIMIC")
result_path = 'D:/project/older_score/result/' # save all results in the folder, the same as 'older_model_main.py'

if (!dir.exists(paste(result_path, "matlab_r/smr/", sep = ""))){
  dir.create(paste(result_path, "matlab_r/smr/", sep = ""))
}

# Part1. SMR models
for(j in db_loop_list){
  data_use <- read.csv(paste(result_path, "model_score/performance_compare/MIMIC_eICU-", j, "/plot_table/smr_plot_model.csv", sep = ""), header=TRUE)
  data_use$gender <- as.factor(data_use$gender)
  colnames(data_use)[which(names(data_use) == "gender")] <- "sex"
  data_use$ethnicity <- as.factor(data_use$ethnicity)
  colnames(data_use)[which(names(data_use) == "ethnicity")] <- "race-ethnicity"
  data_use$age <- as.factor(data_use$age)
  data_use$bmi <- as.factor(data_use$bmi)
  loop_list <- list("lr", "xgb", "rf", "nb") 
  for(i in loop_list){
    x <- SMR.table(data = data_use, obs.var = "true_label", 
                   pred.var = i, 
                   group.var = c("race-ethnicity", "sex", "age")) # "bmi"
    write.csv(x, paste(result_path, "matlab_r/smr/", j, "_smr_", i, ".csv", sep = ""))
    png(file = paste(result_path, "matlab_r/smr/", j, "_smr_", i, ".png", sep = ""), width = 2800, height = 2200, res = 400)
    forest.SMR(x, digits = 2, smr.xlab = "Standardized Mortality Ratio")
    dev.off()
  }
}
rm(data_use, loop_list, x, i, j)


# get scores smr info
for(j in db_loop_list){
  data_use <- read.csv(paste(result_path, "model_score/performance_compare/MIMIC_eICU-", j, "/plot_table/smr_plot_score.csv", sep = ""), header=TRUE)
  data_use$gender <- as.factor(data_use$gender)
  colnames(data_use)[which(names(data_use) == "gender")] <- "sex"
  data_use$ethnicity <- as.factor(data_use$ethnicity)
  colnames(data_use)[which(names(data_use) == "ethnicity")] <- "race-ethnicity"
  data_use$age <- as.factor(data_use$age)
  data_use$bmi <- as.factor(data_use$bmi)
  if (j == 'eICU') {
    loop_list <- list("saps", "oasis", "sofa", "apsiii", "apache_iv")
  }else {
    loop_list <- list("saps", "oasis", "sofa", "apsiii")
  }
  for(i in loop_list){
    x <- SMR.table(data = data_use, obs.var = "true_label", 
                   pred.var = i, 
                   group.var = c("race-ethnicity", "sex", "age")) # "bmi", 
    write.csv(x, paste(result_path, "matlab_r/smr/", j, "_smr_", i, ".csv", sep = ""))
    png(file = paste(result_path, "matlab_r/smr/", j, "_smr_", i, ".png", sep = ""), width = 2800, height = 2200, res = 400)
    forest.SMR(x, digits = 2, smr.xlab = paste("Standardized Mortality Ratio", i, sep = " "))
    dev.off()
  }
}
rm(data_use, loop_list, x, i, j)


# plot smr of eICU XGBoost (individually run)
if (!dir.exists(paste(result_path, "matlab_r/smr_forestplot/", sep = ""))){
  dir.create(paste(result_path, "matlab_r/smr_forestplot/", sep = ""))
}
# save the eicu_smr_xgb as format to plot forestplot
data_need_path = paste(result_path, "matlab_r/smr_forestplot/", sep = "")
rs_forest <- read.csv(paste(data_need_path, "eicu_smr_xgb.csv", sep = ""), header = FALSE)
png(paste(data_need_path, "eicu_smr_xgb_plot.png", sep = ""),height = 1600,width = 3200, res= 300)
forestplot(labeltext = as.matrix(rs_forest[,1:5]),          
           mean = rs_forest$V6,           
           lower = rs_forest$V7,           
           upper = rs_forest$V8,           
           is.summary=c(T,T,T,F,F,T,F,F,T,F,F,F,F),         
           zero = 1,
           boxsize = 0.4,
           lineheight = unit(8,'mm'),
           colgap = unit(2,'mm'),
           lwd.zero = 2,
           lwd.ci = 2,
           col=fpColors(box='#444444',summary="#666666",lines = 'black',zero = '#7AC5CD'),
           txt_gp = fpTxtGp(ticks=gpar(cex=0.6)),           
           xlab="Standardized Mortality Ratio",          
           lwd.xaxis=2,          
           lty.ci = "solid",
           xticks = c(0.5, 0.75, 1, 1.25, 1.5),
           graph.pos = 5
)
rm(rs_forest)