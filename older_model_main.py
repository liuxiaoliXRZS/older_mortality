import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.utils import resample
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC
import csv
import ast
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pyroc
import os
import warnings
warnings.filterwarnings('ignore',
                        '.*',
                        UserWarning,
                        'warnings_filtering')
import re
import shap
from tableone import TableOne
from sklearn.utils import resample
import pickle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore") # stop showing abnormal

import general_utils_data_processing as gudp
import general_utils_eva_fun as guef
import general_utils_model_score_eva as gumse 
import general_utils_plot_table as gupt
import train_tuning_models as ttm
import json

# data and result path setting
data_initial_path = 'D:/project/older_score/data/' # store raw data of multiple sources
result_path = 'D:/project/older_score/result/' # save all results in the folder


#                                   Part 1. data processing |  get the prepared data                               #

# -----------------------------------------------------------------------------------------------------------------#
# aim >>> drop outliers, get tableone info, missing ratio, split data to different usage
# input: mimic.csv, eicu.csv, ams.csv
# output: missing data and tableone info
#         mimic_no_outlier.csv, eicu_no_outlier.csv, ams_no_outlier.csv

data_load_path = ''
data_load_path = data_initial_path # store raw data of multiple sources
outlier_range_check = pd.read_csv(data_load_path + 'outlier_range_check.csv')

if not os.path.exists(data_load_path + 'model_use/no_outlier/'):
    os.makedirs(data_load_path + 'model_use/no_outlier/')
for db_name in ['mimic', 'eicu', 'ams']:
    data_use, data_no_outlier, data_use_name = pd.DataFrame(), pd.DataFrame(), ''
    data_use = pd.read_csv(data_load_path + db_name + '.csv')
    result_path_save = '' # save tableone and missing results
    result_path_save = result_path + 'tableone/'
    if not os.path.exists(result_path_save):
        os.makedirs(result_path_save)
    data_no_outlier = gudp.data_process1_older_score(data_use, db_name, outlier_range_check, result_path_save)
    data_no_outlier.to_csv(data_load_path + 'model_use/no_outlier/' + db_name + '_no_outlier.csv', index=False)
del data_use, data_no_outlier, db_name

# aim >>> obtain the development, external1, external2, temporal datasets
# input: mimic_no_outlier, eicu_no_outlier, ams_no_outlier.csv
# output: MIMIC_use & eICU_use, eICU_extra, ams_use, MIMIC_temporal.csv
data_load_path = ''
data_load_path = data_initial_path + 'model_use/no_outlier/' # store the used data after data processing
MIMIC_initial = pd.read_csv(data_load_path + 'mimic_no_outlier.csv')
eICU_initial = pd.read_csv(data_load_path + 'eicu_no_outlier.csv')
ams_initial = pd.read_csv(data_load_path + 'ams_no_outlier.csv')
MIMIC_eICU_use, MIMIC_temporal, eICU_extra, ams_use  = \
gudp.get_data_diff_source(MIMIC_initial, eICU_initial, ams_initial)

if not os.path.exists(data_initial_path + 'model_use/no_imputation/'):
    os.makedirs(data_initial_path + 'model_use/no_imputation/')
MIMIC_eICU_use.to_csv(data_initial_path + 'model_use/no_imputation/' + 'MIMIC_eICU_use.csv', index=False)
MIMIC_temporal.to_csv(data_initial_path + 'model_use/no_imputation/' + 'MIMIC_temporal.csv', index=False)
eICU_extra.to_csv(data_initial_path + 'model_use/no_imputation/' + 'eICU_extra.csv', index=False)
ams_use.to_csv(data_initial_path + 'model_use/no_imputation/' + 'ams_use.csv', index=False)
del MIMIC_initial, eICU_initial, ams_initial, MIMIC_eICU_use, MIMIC_temporal, eICU_extra, ams_use


#                                   Part 2. different methods to impute data                                   #

# -------------------------------------------------------------------------------------------------------------#
# aim >>> imputation, add new features
# input: MIMIC_eICU_use, MIMIC_temporal, eICU_extra, ams_use.csv
# output: MIMIC_use_imp, MIMIC_temporal_imp, eICU_use_imp, eICU_extra_imp, ams_use_imp.csv
#         imputation  - using missforest

data_load_path = ''
data_load_path = data_initial_path + 'model_use/no_imputation/'# store the used data after data processing

data_save_path = ''
if not os.path.exists(data_initial_path + 'model_use/miceforest_lgb/'):
    os.makedirs(data_initial_path + 'model_use/miceforest_lgb/')
data_save_path = data_initial_path + 'model_use/miceforest_lgb/'

data = {}
for i in ['MIMIC_eICU_use', 'eICU_extra', 'ams_use', 'MIMIC_temporal']:
    data[i] = pd.read_csv(data_load_path + i + '.csv')
data_final = gudp.generate_data_imputation_miceforest(data, data_save_path)


#            Part 3. get the hyperparameters of xgboost choosing different imbalance and imputation methods               #

# ------------------------------------------------------------------------------------------------------------------------#
# aim >>> acquire hyperparameters settings: nonprocess, upsampling, downsampling & miceforest_lgb, ...
# input: MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal.csv
# output: hyperparameter sets

# will run long time, around 14 hours
result_path_save = ''
if not os.path.exists(result_path + 'imbalance_exp/'):
    os.makedirs(result_path + 'imbalance_exp/')
result_path_save = result_path + 'imbalance_exp/'

with open(result_path_save + 'tuning_xgboost.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['imputation_name', 'imbalance_name', 'hyperparam', 'auc', 'sensitivity', 'specificity', 'accuracy', 'F1'])

for m in ['miceforest_lgb']: # if want to try more imputation methods
    train_test_name = 'MIMIC_eICU-MIMIC_eICU'
    data_load_path = ''
    data_load_path = data_initial_path + 'model_use/' + m + '/'  # store the used data after data imputation
    development_set = pd.DataFrame() # evaluation_set: internal validation
    development_set = pd.read_csv(data_load_path + 'MIMIC_eICU_use.csv')
    

    for i in ['nonprocess', 'upsampling', 'downsampling', 'xgboost_setting']:
        X_train_type2, X_train_type2_use, X_eval_type2_use = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        y_train_type2, y_train_type2_use, y_eval_type2_use = pd.Series([]), pd.Series([]), pd.Series([])

        development_set = gudp.diff_data_type(development_set, 2, 'death_hosp', i)
        X_train_type2, y_train_type2 = development_set['X_train'], development_set['y_train']

        X_train_type2_use, X_eval_type2_use, y_train_type2_use, y_eval_type2_use = \
            train_test_split(X_train_type2, y_train_type2, test_size=0.2, random_state=0)
        if i == 'xgboost_setting':
            para_clf, roc_plot_clf, parameters_clf = \
                ttm.train_xgb_model(X_train_type2_use, y_train_type2_use, X_eval_type2_use, y_eval_type2_use, \
                                    development_set['X_test'], development_set['y_test'], 80, 7)
        else:
            para_clf, roc_plot_clf, parameters_clf = \
                ttm.train_xgb_model(X_train_type2_use, y_train_type2_use, X_eval_type2_use, y_eval_type2_use, \
                                    development_set['X_test'], development_set['y_test'], 80, 1)            

        # save the optimal parameters and performance's results
        result_each = []
        result_each = [m, i, parameters_clf, round(para_clf['auc'],3), round(para_clf['sensitivity'],3), \
                       round(para_clf['specificity'],3), round(para_clf['accuracy'],3), round(para_clf['F1'],3)]
        result_all = []
        result_all = open(result_path_save + 'tuning_xgboost.csv', 'a', newline='')
        writer = csv.writer(result_all)
        writer.writerow(result_each)
        result_all.close()


#                    Part4. acquire the internal, external1, external2, temporal performance                    #

# ------------------------------------------------------------------------------------------------------------- #
# aim >>> acquire the performance in 4 types: nonprocess, upsampling, downsampling & miceforest_lgb
# input: MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal.csv
# output: xbg_no_cal.dat, xgb_cal.dat, feature rankings.csv/png, pred_real_info.csv (probability)
#         imputation_imbalance_perf_95ci.csv

params_all = pd.read_csv(result_path + 'imbalance_exp/' + 'tuning_xgboost.csv')

for imputation_med in ['miceforest_lgb']: # if want to try more imputation methods
    data_load_path = ''
    data_load_path = data_initial_path + 'model_use/' + imputation_med + '/'  # store the used data after data imputation
    MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    MIMIC_eICU_use = pd.read_csv(data_load_path + 'MIMIC_eICU_use.csv')
    eICU_extra = pd.read_csv(data_load_path + 'eICU_extra.csv')
    ams_use = pd.read_csv(data_load_path + 'ams_use.csv')
    MIMIC_temporal = pd.read_csv(data_load_path + 'MIMIC_temporal.csv')

    for imbalance_med in ['upsampling', 'downsampling', 'xgboost_setting', 'nonprocess']:

        for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']: #
            result_path_save = ''
            result_path_save = result_path + 'imbalance_exp/' + imputation_med + '/' +  imbalance_med + '/' + train_test_name + '/'
            if not os.path.exists(result_path_save):
                os.makedirs(result_path_save)

            model_use_yn = ''
            if train_test_name == 'MIMIC_eICU-MIMIC_eICU':
                model_use_yn = 'False'
            else:
                model_use_yn = 'True'

            # [1] get the different usage data
            data_need = {}
            data_need = gudp.get_train_cal_test_data(MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal, train_test_name, imbalance_med)
            model_info = {
                'model_use':{
                    'existing': model_use_yn, \
                    'params': ast.literal_eval(params_all.loc[(params_all['imputation_name'] == imputation_med) & (params_all['imbalance_name'] == imbalance_med), 'hyperparam'].values[0])
                             },
                'model_path_full_info': result_path + 'imbalance_exp/' + imputation_med + '/' +  imbalance_med + '/' + 'MIMIC_eICU-MIMIC_eICU' + '/' + 'xgb_no_cal.dat',
                'ths_use': 'False',
                'ths_value': 0,
                'cal_yn': 'True',
                'shap_info':{'shap_yn': 'False', 'image_full_info': result_path_save + 'features_ranking.png', \
                             'fea_table_full_info': result_path_save + 'features_ranking.csv'},
                'no_cal_model_full_info': result_path_save + 'xgb_no_cal.dat',
                'cal_model_full_info': result_path_save + 'xgb_cal.dat'
            }
            model_info['model_use']['params'].pop('missing', None) # existing error if using missing

            result_info = {}
            result_info = gumse.older_xgb_model(data_need['data_need_type2'], [], model_info)
            result_info['pred_real_info'][['id', 'true_label']] = result_info['pred_real_info'][['id', 'true_label']].astype('Int64')
            result_info['pred_real_info'].to_csv(result_path_save + 'pred_real_info.csv', index=False)


# calculate the performance using bootstrap with more metrics
# focus on roc_auc and ap, sensitivity and specificity, ... can be used to compare the performance
with open(result_path + 'imbalance_exp/' + 'imputation_imbalance_perf_95ci.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['imputation_med', 'imbalance_med', 'train_test_name', \
                    'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'])

for imputation_med in ['miceforest_lgb']: # 'median_indicator', 
    for imbalance_med in ['downsampling', 'xgboost_setting', 'nonprocess', 'upsampling']:
        for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
            data_load_path = ''
            data_load_path = result_path + 'imbalance_exp/' + imputation_med + '/' + imbalance_med + '/' + train_test_name + '/'
            pred_real_info = pd.read_csv(data_load_path + 'pred_real_info.csv')[['true_label', 'xgb']]
            pred_real_info.rename(columns={'xgb':'probability'}, inplace = True)

            # acquire the threshold
            parameters_need, result_each = {}, []
            parameters_need, _ = guef.model_performance_params(pred_real_info['true_label'], pred_real_info['probability'], 'False', 0)
            _, result_each = guef.model_performance_params_bootstrap_95CI(pred_real_info, parameters_need['threshold'], 500)
            result_each = [imputation_med, imbalance_med, train_test_name] + result_each.values.flatten().tolist()
            result_all = []
            result_all = open(result_path + 'imbalance_exp/' + 'imputation_imbalance_perf_95ci.csv', 'a', newline='')
            writer = csv.writer(result_all)
            writer.writerow(result_each)
            result_all.close()


# according to analyze the result (balance the AUROC and AUPRC), we decided to choose the 'miceforest imputation' + 'nonprocess'


#                    Part 5. compare with multiple ML and clinical scores                    #

#------------------------------------------------------------------------------------------- #
# aim >>> obtain all performance metrics of models and clinical scores, and subgroup performance based on the Part 4
# input: MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal.csv
# output: in each study set existing
#         models_perf_matric.csv, models_pred_true.csv, scores_perf_matric.csv, scores_pred_true.csv
#         modelname_cal.dat,  modelname_no_cal.dat
#         features_ranking.png, features_ranking.csv - xgboost - (MIMIC_eICU-MIMIC_eICU)
#         roc_models_scores_set.json
#   plot_table subfolder:
#         roc_model_score_compare.png
#         model_score_perf_matric_95ci.csv, model_score_perf_matric_bts.csv
#         sub_model_score_perf_matric_95ci.csv, sub_model_score_perf_matric_bts.csv

# [0] prepare the basic information: data, result_path_save, and selected hyperparameters | change all names
data_load_path = ''
data_load_path = data_initial_path + 'model_use/miceforest_lgb/'  # store the used data after data using miceforest_lgb
columns_name = pd.read_csv(data_initial_path + 'columns_name.csv')
MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
MIMIC_eICU_use = pd.read_csv(data_load_path + 'MIMIC_eICU_use.csv')
MIMIC_eICU_use.columns = columns_name['new_name']
eICU_extra = pd.read_csv(data_load_path + 'eICU_extra.csv')
eICU_extra.columns = columns_name['new_name']
ams_use = pd.read_csv(data_load_path + 'ams_use.csv')
ams_use.columns = columns_name['new_name']
MIMIC_temporal = pd.read_csv(data_load_path + 'MIMIC_temporal.csv')
MIMIC_temporal.columns = columns_name['new_name']
# merge clinical scores ['apsiii', 'sofa', 'oasis', 'saps']
score_all, scores_list = pd.DataFrame(), []
scores_list = ['apsiii', 'sofa', 'oasis', 'saps', 'oasis_prob', 'apsiii_prob', 'saps_prob', 'sofa_prob']
score_all = pd.read_csv(data_initial_path + 'mimic.csv')[scores_list + ['id']]
score_all = score_all.append(pd.read_csv(data_initial_path + 'eicu.csv')[scores_list + ['id']])
score_all = score_all.append(pd.read_csv(data_initial_path + 'ams.csv')[scores_list + ['id']])
score_all.reset_index(drop=True, inplace=True)
MIMIC_eICU_use = MIMIC_eICU_use.merge(score_all, how='left', on='id')
eICU_extra = eICU_extra.merge(score_all, how='left', on='id')
ams_use = ams_use.merge(score_all, how='left', on='id')
MIMIC_temporal = MIMIC_temporal.merge(score_all, how='left', on='id')
del score_all, scores_list

result_path_save = ''
if not os.path.exists(result_path + 'model_score/'):
    os.makedirs(result_path + 'model_score/')
if not os.path.exists(result_path + 'model_score/performance_compare/'):
    os.makedirs(result_path + 'model_score/performance_compare/')
result_path_save = result_path + 'model_score/performance_compare/'

parameters_using_optimal, params_all = {}, pd.DataFrame()
params_all = pd.read_csv(result_path + 'imbalance_exp/tuning_xgboost.csv')
parameters_using_optimal = ast.literal_eval(params_all.loc[(params_all['imputation_name'] == 'miceforest_lgb') & (params_all['imbalance_name'] == 'nonprocess'), 'hyperparam'].values[0])
parameters_using_optimal.pop('missing', None) # existing error if containing missing
del params_all


# [1] get the N models and M scores' performance
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    # (0) get the different usage data
    data_need = {}
    data_need = gudp.get_train_cal_test_data(MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal, train_test_name, 'nonprocess')

    # (1) get the performance of ML models
    path_save = ''
    path_save = result_path_save + train_test_name + '/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    model_use_set = {}
    model_use_set = {'MIMIC_eICU-MIMIC_eICU':'False', 'MIMIC_eICU-eICU':'True', \
                    'MIMIC_eICU-Ams':'True', 'MIMIC_eICU-MIMIC':'True'} # 'True': use the formal generated
    # just set the xgb model
    threshold_use_set = {'MIMIC_eICU-MIMIC_eICU':{
                                                    'lr':{'ths_use': 'True', 'ths_value': 0.07},
                                                    'svm':{'ths_use': 'True', 'ths_value': 0.065},
                                                    'xgb':{'ths_use': 'True', 'ths_value': 0.10},
                                                    'rf':{'ths_use': 'True', 'ths_value': 0.112},
                                                    'nb':{'ths_use': 'True', 'ths_value': 0.131}
                                                   }, \
                           'MIMIC_eICU-eICU':{
                                              'lr':{'ths_use': 'True', 'ths_value': 0.1},
                                              'svm':{'ths_use': 'True', 'ths_value': 0.099},
                                              'xgb':{'ths_use': 'True', 'ths_value': 0.098},
                                              'rf':{'ths_use': 'True', 'ths_value': 0.082},
                                              'nb':{'ths_use': 'True', 'ths_value': 0.119}
                                             }, \
                           'MIMIC_eICU-Ams':{
                                             'lr':{'ths_use': 'True', 'ths_value': 0.095},
                                             'svm':{'ths_use': 'False', 'ths_value': 0},
                                             'xgb':{'ths_use': 'True', 'ths_value': 0.12},
                                             'rf':{'ths_use': 'False', 'ths_value': 0},
                                             'nb':{'ths_use': 'True', 'ths_value': 0.13}
                                            }, \
                           'MIMIC_eICU-MIMIC':{
                                               'lr':{'ths_use': 'False', 'ths_value': 0},
                                               'svm':{'ths_use': 'True', 'ths_value': 0.03},
                                               'xgb':{'ths_use': 'False', 'ths_value': 0},
                                               'rf':{'ths_use': 'False', 'ths_value': 0},
                                               'nb':{'ths_use': 'False', 'ths_value': 0}
                                              }
                        }

    all_models_info = {'lr':{'model_use': model_use_set[train_test_name], 'cal_yn': 'True',
                               'no_cal_model_full_info': path_save + 'lr_no_cal.dat',
                               'cal_model_full_info': path_save + 'lr_cal.dat',
                               'model_path_full_info': result_path_save + 'MIMIC_eICU-MIMIC_eICU/' + 'lr_no_cal.dat',
                               'ths_use': threshold_use_set[train_test_name]['lr']['ths_use'],
                               'ths_value': threshold_use_set[train_test_name]['lr']['ths_value']
                              },
                        'svm':{'model_use': model_use_set[train_test_name], 'cal_yn': 'True',
                               'no_cal_model_full_info': path_save + 'svm_no_cal.dat',
                               'cal_model_full_info': path_save + 'svm_cal.dat',
                               'model_path_full_info': result_path_save + 'MIMIC_eICU-MIMIC_eICU/' + 'svm_no_cal.dat',
                               'ths_use': threshold_use_set[train_test_name]['svm']['ths_use'],
                               'ths_value': threshold_use_set[train_test_name]['svm']['ths_value']
                              },
                        'xgb':{'model_use':{'existing': model_use_set[train_test_name], 'params': parameters_using_optimal},
                               'cal_yn': 'True',
                               'no_cal_model_full_info': path_save + 'xgb_no_cal.dat',
                               'cal_model_full_info': path_save + 'xgb_cal.dat',
                               'shap_info':{'shap_yn': 'True', 'image_full_info': path_save + 'features_ranking.png',
                                            'image_full_info_bar': path_save + 'features_ranking_bar.png',
                                            'fea_table_full_info': path_save + 'features_ranking.csv'
                                           },
                               'model_path_full_info': result_path_save + 'MIMIC_eICU-MIMIC_eICU/' + 'xgb_no_cal.dat',
                               'ths_use': threshold_use_set[train_test_name]['xgb']['ths_use'],
                               'ths_value': threshold_use_set[train_test_name]['xgb']['ths_value']
                              },
                        'rf':{'model_use': model_use_set[train_test_name], 'cal_yn': 'True',
                              'no_cal_model_full_info': path_save + 'rf_no_cal.dat',
                              'cal_model_full_info': path_save + 'rf_cal.dat',
                              'model_path_full_info': result_path_save + 'MIMIC_eICU-MIMIC_eICU/' + 'rf_no_cal.dat',
                              'ths_use': threshold_use_set[train_test_name]['rf']['ths_use'],
                              'ths_value': threshold_use_set[train_test_name]['rf']['ths_value']
                              },
                        'nb':{'model_use': model_use_set[train_test_name], 'cal_yn': 'True',
                              'no_cal_model_full_info': path_save + 'nb_no_cal.dat',
                              'cal_model_full_info': path_save + 'nb_cal.dat',
                              'model_path_full_info': result_path_save + 'MIMIC_eICU-MIMIC_eICU/' + 'nb_no_cal.dat',
                              'ths_use': threshold_use_set[train_test_name]['nb']['ths_use'],
                              'ths_value': threshold_use_set[train_test_name]['nb']['ths_value']
                              }
                      }

    threshold_scores_use_set = {
        'MIMIC_eICU-MIMIC_eICU':{
            'apsiii':{'ths_use': 'True', 'ths_value': 0.11},
            'sofa':{'ths_use': 'True', 'ths_value': 0.11},
            'oasis':{'ths_use': 'False', 'ths_value': 0},
            'saps':{'ths_use': 'False', 'ths_value': 0}
            }, \
        'MIMIC_eICU-eICU':{
            'apsiii':{'ths_use': 'True', 'ths_value': 0.10},
            'sofa':{'ths_use': 'True', 'ths_value': 0.123},
            'oasis':{'ths_use': 'True', 'ths_value': 0.11},
            'saps':{'ths_use': 'True', 'ths_value': 0.35},
            'apache_iv':{'ths_use': 'True', 'ths_value': 0.13}
            }, \
        'MIMIC_eICU-Ams':{
            'apsiii':{'ths_use': 'True', 'ths_value': 0.12},
            'sofa':{'ths_use': 'True', 'ths_value': 0.11},
            'oasis':{'ths_use': 'False', 'ths_value': 0},
            'saps':{'ths_use': 'True', 'ths_value': 0.45}
            }, \
        'MIMIC_eICU-MIMIC':{
            'apsiii':{'ths_use': 'True', 'ths_value': 0.1},
            'sofa':{'ths_use': 'True', 'ths_value': 0.11},
            'oasis':{'ths_use': 'False', 'ths_value': 0},
            'saps':{'ths_use': 'False', 'ths_value': 0}
            }
    }

    roc_models_set = {}
    roc_models_set = gumse.performance_machine_learning_models(data_need['data_need_type1'], data_need['data_need_type2'], \
                                                               path_save=path_save, \
                                                               all_models_info=all_models_info
                                                               )
    roc_scores_set = {}
    roc_scores_set = gumse.performance_scores(data_need['data_need_type2'], train_test_name=train_test_name, \
        path_save=path_save, all_scores_info=threshold_scores_use_set)

    # save results
    roc_models_scores_set = {}
    roc_models_scores_set = {'roc_models_set':roc_models_set, 'roc_scores_set':roc_scores_set}
    with open(path_save + 'roc_models_scores_set.json', "w") as outfile:
        json.dump(roc_models_scores_set, outfile)

# plot models and scores roc curves comparsion
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']: 
    data_load_path = result_path_save + train_test_name + '/'
    path_save = data_load_path + 'plot_table/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # plot models and scores roc comparison
    roc_models_scores_set = {}
    with open(data_load_path + 'roc_models_scores_set.json') as json_file:
        roc_models_scores_set = json.load(json_file)
    gupt.roc_models_scores_compare(roc_data_set=roc_models_scores_set, train_test_name=train_test_name, path_save=path_save)

# result_path_save = result_path + 'model_score/performance_compare/' # !!!extra adding

# get the 95% CI results of models and scores
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = result_path_save + train_test_name + '/'
    path_save = data_load_path + 'plot_table/'

    # load data
    data_pred_true, data_threshold = pd.DataFrame(), pd.DataFrame()
    data_model = pd.read_csv(data_load_path + 'models_pred_true.csv')
    threshold_set_model = pd.read_csv(data_load_path + 'models_perf_matric.csv')
    data_score = pd.read_csv(data_load_path + 'scores_pred_true.csv')
    threshold_set_score = pd.read_csv(data_load_path + 'scores_perf_matric.csv')
    # [2]. get the 95% CI of models and scores of total cohorts
    gupt.model_score_perf_matric_95ci(data_model=data_model, threshold_set_model=threshold_set_model, \
                                    data_score=data_score, threshold_set_score=threshold_set_score, \
                                    train_test_name=train_test_name, path_save=path_save)
    # [3]. get the sub-population performance of 95% CI
    gupt.sub_model_score_perf_matric_95ci(data_model=data_model, threshold_set_model=threshold_set_model, \
                                        data_score=data_score, threshold_set_score=threshold_set_score, \
                                        train_test_name=train_test_name, path_save=path_save)


#                                   Part 6 get xgb performance of part of features and save the roc results                                    #
#
# -------------------------------------------------------------------------------------------------------------------------------------------- #
# aim >>> obtain the performance using part of features [5, 10, ..., 60]
# input: features_ranking.csv (MIMIC_eICU-MIMIC_eICU) | MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal
# output: in each study set existing
#         all_xgb_perf_matric.csv, all_xgb_pred_true.csv
#         partn_xgb_cal.dat, partn_xgb_no_cal.dat, partn_features_ranking.csv, partn_features_ranking.png (MIMIC_eICU-MIMIC_eICU)
#   plot_table subfolder:
#         part_fea_perf_matric_95ci.csv, part_fea_perf_matric_bts.csv

data_load_path = data_initial_path + 'model_use/miceforest_lgb/'
columns_name = pd.read_csv(data_initial_path + 'columns_name.csv')
MIMIC_eICU_use = pd.read_csv(data_load_path + 'MIMIC_eICU_use.csv')
MIMIC_eICU_use.columns = columns_name['new_name']
eICU_extra = pd.read_csv(data_load_path + 'eICU_extra.csv')
eICU_extra.columns = columns_name['new_name']
ams_use = pd.read_csv(data_load_path + 'ams_use.csv')
ams_use.columns = columns_name['new_name']
MIMIC_temporal = pd.read_csv(data_load_path + 'MIMIC_temporal.csv')
MIMIC_temporal.columns = columns_name['new_name']

fea_num_info = {}
fea_num_info = {'num': [5, 10, 15, 20, 25, 30, 40, 50, 60], \
                'default_use': ['True', 'True', 'True', 'False', 'False', \
                                'False', 'False', 'False', 'False'
                                ]
                }

path_use = result_path + 'model_score/performance_compare/MIMIC_eICU-MIMIC_eICU/'
feature_ranking = pd.read_csv(path_use + 'features_ranking.csv')
del path_use

# 2. get the different aims data
if not os.path.exists(result_path + 'model_score/part_fea/'):
    os.makedirs(result_path + 'model_score/part_fea/')
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_need = {}
    data_need = gudp.get_train_cal_test_data(MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal, train_test_name, 'nonprocess')

    #                     Part 6.1 get performance of models and scores and save the roc results                    #
    path_save = ''
    path_save = result_path + 'model_score/part_fea/' + train_test_name + '/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    model_use_set = {}
    model_use_set = {'MIMIC_eICU-MIMIC_eICU':'False', 'MIMIC_eICU-eICU':'True', \
                     'MIMIC_eICU-Ams':'True', 'MIMIC_eICU-MIMIC':'True'} # 'True': use the formal generated
    # save results
    all_models_perf_matric, all_models_pred_true = [], pd.DataFrame()

    for i in range(len(fea_num_info['num'])):
        fea_num, default_use= 0, ''
        fea_num, default_use = fea_num_info['num'][i], fea_num_info['default_use'][i]

        model_info = {}
        model_info = {'model_use': model_use_set[train_test_name], \
                      'default_use': default_use, \
                       'cal_yn': 'True', 'no_cal_model_full_info': path_save + 'part' + str(fea_num) + '_xgb_no_cal.dat', \
                       'cal_model_full_info': path_save + 'part' + str(fea_num) + '_xgb_cal.dat', \
                       'shap_info':{'shap_yn': 'False', 'image_full_info': path_save + 'part' + str(fea_num) + '_features_ranking.png', \
                                        'fea_table_full_info': path_save + 'part' + str(fea_num) + '_features_ranking.csv'
                                   }, \
                       'model_path_full_info': result_path + 'model_score/part_fea/MIMIC_eICU-MIMIC_eICU/part' + str(fea_num) + '_xgb_no_cal.dat', \
                       'ths_use': 'False', 'ths_value': 0
                      }

        drop_cols_name = [] # clinical scores and expect the top fea_num features
        # drop_cols_name = ['apsiii', 'apsiii_prob', 'oasis', 'oasis_prob', 'saps', 'saps_prob', 'sofa', 'sofa_prob']
        drop_cols_name = drop_cols_name + list(feature_ranking.iloc[fea_num:]['col_name'])
        roc_models_set = {}
        roc_models_set = gumse.older_xgb_part_fea_model(data_need['data_need_type2'], drop_cols_name, model_info)

        models_perf_matric, models_pred_true = [], pd.DataFrame()
        models_perf_matric = roc_models_set['perf_matric']
        models_perf_matric = ['xgb_' + str(fea_num)] + models_perf_matric[1:]
        all_models_perf_matric.append(models_perf_matric)
        models_pred_true = roc_models_set['pred_real_info']
        models_pred_true.rename(columns={'xgb': 'xgb_'+str(fea_num)}, inplace=True)
        if all_models_pred_true.shape[0] == 0:
            all_models_pred_true = models_pred_true
        else:
            all_models_pred_true = pd.merge(all_models_pred_true, models_pred_true, on=['id', 'true_label'])
    all_models_perf_matric = pd.DataFrame(all_models_perf_matric)
    all_models_perf_matric.columns = ['name', 'roc_auc', 'sensitivity', 'specificity', \
                                      'accuracy', 'F1', 'precision', 'ap', 'brier_score', 'threshold'
                                      ]
    all_models_perf_matric.to_csv(path_save + 'all_xgb_perf_matric.csv', index=False)
    all_models_pred_true.to_csv(path_save + 'all_xgb_pred_true.csv', index=False)


for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = result_path + 'model_score/part_fea/' + train_test_name + '/'
    path_save = ''
    path_save = data_load_path + 'plot_table/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # load data
    data_model = pd.read_csv(data_load_path + 'all_xgb_pred_true.csv')
    threshold_set_model = pd.read_csv(data_load_path + 'all_xgb_perf_matric.csv')
    gupt.part_fea_perf_matric_95ci(data=data_model, threshold_set=threshold_set_model, path_save=path_save)


#                                 Part 7. matlab plot needed data                                     #

# --------------------------------------------------------------------------------------------------- #
# aim >>> get matlab plot data
# input: model_score_perf_matric_95ci.csv
# output:
#           models_scores_95CI_auc.csv, auc_all_subgroup_95CI_plot.csv
#           roc_auc_pvalue.xlsx, ap_pvalue.xlsx (data_load_path)
#           smr_plot_model.csv, smr_plot_score.csv (data_load_path)
#           part_fea_models_auc_all_95CI_plot.csv
#           dbname_drop_fea_models_compare_diff_auc.txt (drop_feature)
#           development|external_1|external_2|temporal_imputation (tableone)
#           development|external_1|external_2|temporal_missing (tableone)
#           features_ranking_round.csv

# auroc comparison of models and scores
data_final = pd.DataFrame()
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = result_path + 'model_score/performance_compare/' + train_test_name + '/plot_table/'
    data, data_use = pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(data_load_path + 'model_score_perf_matric_95ci.csv')
    data_use = gupt.generate_matplab_plot_95ci_split(data)
    data_use['db_source'] = train_test_name.replace('MIMIC_eICU-', '')
    data_final = data_final.append(data_use)
if not os.path.exists(result_path + 'matlab_r/'):
    os.makedirs(result_path + 'matlab_r/')
data_final.to_csv(result_path + 'matlab_r/' + 'models_scores_95CI_auc.csv', index=False)

# get models and scores' comparison results of p value
for test_set_name in ['MIMIC_eICU', 'eICU', 'Ams', 'MIMIC']:
    data_load_path = result_path + 'model_score/performance_compare/MIMIC_eICU-' + test_set_name + '/plot_table/'
    data = pd.read_csv(data_load_path + 'model_score_perf_matric_bts.csv')
    gupt.auc_ap_model_score_vs_pvalue(data, test_set_name, data_load_path)

data_final = pd.DataFrame()
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = result_path + 'model_score/performance_compare/' + train_test_name + '/plot_table/'
    data, data_use = pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(data_load_path + 'sub_model_score_perf_matric_95ci.csv')
    data_use = gupt.sub_auc_xgb_plot_95ci_split(data)
    data_use['db_source'] = train_test_name.replace('MIMIC_eICU-', '')
    data_final = data_final.append(data_use)
data_final.to_csv(result_path + 'matlab_r/' + 'auc_all_subgroup_95CI_plot.csv', index=False)

# get r plot smr needed info of models
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = result_path + 'model_score/performance_compare/' + train_test_name + '/'
    data, data_use = pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(data_load_path + 'models_pred_true.csv')
    data_use = gupt.basic_info_merge(data)
    data_use = data_use.loc[~(data_use['ethnicity'] == 'other')].reset_index(drop=True)
    data_use = data_use[['id', 'true_label', 'lr', 'svm', 'xgb', 'rf', 'nb', 'age_group', \
                        'bmi_group', 'gender_group', 'ethnicity_group']]
    data_use.rename(columns={'age_group': 'age', 'bmi_group': 'bmi', 'gender_group': 'gender', 'ethnicity_group': 'ethnicity'}, inplace=True)
    data_use.to_csv(data_load_path + 'plot_table/' + 'smr_plot_model.csv', index=False)

# get r plot smr needed info of scores
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = result_path + 'model_score/performance_compare/' + train_test_name + '/'
    data, data_use = pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(data_load_path + 'scores_pred_true.csv')
    data_use = gupt.basic_info_merge(data)
    data_use = data_use.loc[~(data_use['ethnicity'] == 'other')].reset_index(drop=True)
    if train_test_name == 'MIMIC_eICU-eICU':
        data_use = data_use[['id', 'true_label', 'oasis', 'apsiii', 'saps', 'sofa', 'apache_iv', 'age_group', \
                            'bmi_group', 'gender_group', 'ethnicity_group']]
    else:
        data_use = data_use[['id', 'true_label', 'oasis', 'apsiii', 'saps', 'sofa', 'age_group', \
                            'bmi_group', 'gender_group', 'ethnicity_group']]
    data_use.rename(columns={'age_group': 'age', 'bmi_group': 'bmi', 'gender_group': 'gender', 'ethnicity_group': 'ethnicity'}, inplace=True)
    data_use.to_csv(data_load_path + 'plot_table/' + 'smr_plot_score.csv', index=False)


# get part of features performance for matlab plot
data_final = pd.DataFrame()
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data_load_path = ''
    data_load_path = result_path + 'model_score/part_fea/' + train_test_name + '/plot_table/'
    data, data_use = pd.DataFrame(), pd.DataFrame()
    data = pd.read_csv(data_load_path + 'part_fea_perf_matric_95ci.csv')
    data_use = gupt.part_fea_auc_xgb_plot_95ci_split(data)
    data_use['db_source'] = train_test_name.replace('MIMIC_eICU-', '')
    data_final = data_final.append(data_use)
data_final.to_csv(result_path + 'matlab_r/' + 'part_fea_models_auc_all_95CI_plot.csv', index=False)


# get tableone info of the development, evaluation, and temporal set
data_load_path = data_initial_path
path_save = result_path + 'tableone/'
gupt.dev_ext_temp_tableone(data_load_path, path_save)

# get missing ratio of the development, evaluation, and temporal set
data_load_path = data_initial_path
path_save = result_path + 'tableone/'
gupt.dev_ext_temp_missing_ratio(data_load_path, path_save)

# get the calibration curves of models and scores
data_load_path = result_path + 'model_score/performance_compare/'
path_save = data_load_path
for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    gupt.older_calibration_curves_set(data_load_path, train_test_name, path_save)


#                                     Part 8. runing code on R                                        #

# --------------------------------------------------------------------------------------------------- #
# aim >>> get smr plot and shap case examples
print('*********************************************************')
print('Please run calibration_older_1.R')
print('*********************************************************')

input("Press Enter to continue if you finish running the R code ...")


#                                   Part 9. compare with GOSSIS and SMS-ICU                            #

# ----------------------------------------------------------------------------------------------------- #
# aim >>> get GOSSIS and SMS-ICU scores' performance
# input:
# output:
path_save = result_path + 'gossis_sms/'
if not os.path.exists(path_save):
    os.makedirs(path_save)

# add id column to gossis data
gossis_need, id_need = pd.DataFrame(), pd.DataFrame()
gossis_need = pd.read_csv(data_initial_path + 'gossis_eicu_db/' + 'gossis-1-eicu-predictions_new.csv')
id_need = pd.read_csv(data_initial_path + 'sms_score/' + 'eicu_sms.csv')[['id', 'patientunitstayid']]
gossis_need.rename(columns={'gossis1_ihm_pred': 'gossis'}, inplace=True)
gossis_need = pd.merge(gossis_need, id_need, on='patientunitstayid')
gossis_need.to_csv(data_initial_path + 'gossis_eicu_db/' + 'eicu_gossis.csv',index=False)
del gossis_need, id_need

# acquire the merged data ['xgb', 'gossis', 'sms']
xgb_path = result_path + 'model_score/performance_compare/' # xgb probability save path
gossis_path = data_initial_path + 'gossis_eicu_db/' # gossis probability save path
sms_path = data_initial_path + 'sms_score/' # sms probability save path
MIMIC_eICU_need, eICU_need, Ams_need, MIMIC_need = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
gudp.generate_xgb_gossis_sms(xgb_path, gossis_path, sms_path, path_save)

data_load_path = ''
data_load_path = path_save

for train_test_name in ['MIMIC_eICU-MIMIC_eICU', 'MIMIC_eICU-eICU', 'MIMIC_eICU-Ams', 'MIMIC_eICU-MIMIC']:
    data = pd.read_csv(data_load_path + train_test_name.replace('MIMIC_eICU-', '') + '_need.csv')
    gupt.sub_xgb_gossis_sms_perf_matric_95ci(data, train_test_name, path_save)


#                                      Part 9. runing code on R                                       #

# --------------------------------------------------------------------------------------------------- #
# aim >>> get smr results of gossis and sms-icu
print('*********************************************************')
print('Please run calibration_older_2.R')
print('*********************************************************')

input("Press Enter to continue if you finish running the R ...")


#                                   Part 11. extra analysis                                     #

# --------------------------------------------------------------------------------------------- #
# acquire all smr with 95% CI results
data_load_path = ''
data_load_path = result_path + 'matlab_r/smr/'
data = pd.DataFrame()
for i in ['MIMIC_eICU', 'eICU', 'Ams', 'MIMIC']:
    for j in ['xgb', 'rf', 'lr', 'nb', 'apsiii', 'oasis', 'saps', 'apache_iv', 'sofa']:
        if not os.path.exists(data_load_path + i + '_smr_' + j + '.csv'):
            continue
        data_each = pd.DataFrame()
        data_each = pd.read_csv(data_load_path + i + '_smr_' + j + '.csv')
        data_each.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_each['db_name'] = i
        data_each['model_name'] = j
        data = data.append(data_each)
data.reset_index(drop=True, inplace=True)
data['smr_95ci'] = round(data['SMR'],3).map(str) + ' (' + round(data['lower.Cl'],3).map(str) + '-' + round(data['upper.Cl'],3).map(str) + ')'
data.to_csv(data_load_path + 'all_smr_merge.csv', index=False)

# gossis and sms-icu
data_load_path = ''
data_load_path = result_path + '/gossis_sms/smr/'
data = pd.DataFrame()
for i in ['MIMIC_eICU', 'eICU', 'Ams', 'MIMIC']:
    for j in ['gossis', 'sms_prob']:
        if not os.path.exists(data_load_path + i + '_smr_' + j + '.csv'):
            continue
        data_each = pd.DataFrame()
        data_each = pd.read_csv(data_load_path + i + '_smr_' + j + '.csv')
        data_each.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_each['db_name'] = i
        data_each['model_name'] = j
        data = data.append(data_each)
data.reset_index(drop=True, inplace=True)
data['smr_95ci'] = round(data['SMR'],3).map(str) + ' (' + round(data['lower.Cl'],3).map(str) + '-' + round(data['upper.Cl'],3).map(str) + ')'
data.to_csv(data_load_path + 'all_smr_merge.csv', index=False)



print('Ending!')