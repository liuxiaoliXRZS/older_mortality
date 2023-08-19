# Step 1. Run all codes in the folder 'bigquery_data_extraction'
#            Please note that we used the MIMIC-III (v1.4), MIMIC-IV (v1.0), eICU-CRD (v1.2), Amsterdam-UMC (v0.1)
#            If you lack some basic codes, you need to look at the following links:
#                   https://github.com/MIT-LCP/mimic-code
#                   https://github.com/MIT-LCP/eicu-code
#                   https://github.com/AmsterdamUMC/AmsterdamUMCdb      
1. Please change 'db_name' to your project and database name
2. Run oasis_firstday_ams.sql  -> sapsii_firstday_ams.sql -> older_score_ams.sql
3. Run sofa_eicu -> charlson_comorbidity_eicu.sql -> older_score_eicu.sql
4. Run older_score_mimiciv  -> charlson_mimiciii.sql -> older_score_mimiciii.sql
5. Run sms_icu_score_all.sql


# Step 2. Data proprocessing, Model development and evaluation
#             Please note that you need to change the paths where data and results are saved
#             
#                           

1. Please change the setting:
(1) older_model_main.py 
data_initial_path = 'D:/project/older_score/data/'
result_path = 'D:/project/older_score/result/'
(2) general_utils_data_processing.py
Function: get_data_diff_source  -> path (should the same as data_initial_path)
Function: generate_xgb_gossis_sms  -> data_basic_path
(3) general_utils_model_score_eva.py
Function: performance_scores -> apache_iv_info
(4) general_utils_plot_table.py
Function: basic_info_merge -> data_each
(5) calibration_older_1/2.R
result_path
2. Run older_model_main.py
#           calibration_older_1.R
#           calibration_older_2.R
