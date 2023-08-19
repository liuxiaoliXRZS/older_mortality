library(ems)
library(dplyr)
library(forestplot)


# gossis, sms-icu
db_loop_list <- list("MIMIC_eICU", "eICU", "Ams", "MIMIC")
result_path = '' # save all results in the folder, the same as 'older_model_main.py'

if (!dir.exists(paste(result_path, "gossis_sms/smr/", sep = ""))){
  dir.create(paste(result_path, "gossis_sms/smr/", sep = ""))
}

for(j in db_loop_list){
  data_use <- read.csv(paste(result_path, "gossis_sms/", j, "_need.csv", sep = ""), header=TRUE)
  data_use$age <- NULL
  data_use$gender_group <- as.factor(data_use$gender_group)
  colnames(data_use)[which(names(data_use) == "gender_group")] <- "sex"
  data_use$ethnicity <- as.factor(data_use$ethnicity)
  colnames(data_use)[which(names(data_use) == "ethnicity")] <- "race-ethnicity"
  data_use$age_group <- as.factor(data_use$age_group)
  colnames(data_use)[which(names(data_use) == "age_group")] <- "age"  
  if (j == 'eICU') {
    loop_list <- list("sms_prob", "gossis")
  }else {
    loop_list <- list("sms_prob")
  }
  for(i in loop_list){
    x <- SMR.table(data = data_use, obs.var = "true_label", 
                   pred.var = i, 
                   group.var = c("race-ethnicity", "sex", "age"))
    write.csv(x, paste(result_path, "gossis_sms/smr/", j, "_smr_", i, ".csv", sep = ""))
    png(file = paste(result_path, "gossis_sms/smr/", j, "_smr_", i, ".png", sep = ""), width = 2800, height = 2200, res = 400)
    forest.SMR(x, digits = 2, smr.xlab = "Standardized Mortality Ratio")
    dev.off()
  }
}
rm(data_use, loop_list, x, i, j)