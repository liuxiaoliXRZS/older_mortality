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
import train_tuning_models as ttm
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pyroc
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
warnings.filterwarnings("ignore") # 拦截异常
import general_utils_data_processing as gudp
import general_utils_eva_fun as guef
import general_utils_model_score_eva as gumse
import json
import gc


def roc_models_scores_compare(roc_data_set, train_test_name, path_save):
    roc_models_set, roc_scores_set = {}, {}
    roc_models_set, roc_scores_set = roc_data_set['roc_models_set'], roc_data_set['roc_scores_set'] 
    plt.rcParams['figure.dpi'] = 600 #分辨率
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['figure.figsize'] = (10.0, 4.3) # 设置figure_size尺寸

    # XGBOOST, LR, RF, SVM, NB
    plt.subplot(1,2,1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(roc_models_set['LR'][0], roc_models_set['LR'][1],
            'g-', label='LR (AUC = %0.3f)' % auc(roc_models_set['LR'][0], roc_models_set['LR'][1]))
    plt.plot(roc_models_set['SVM'][0], roc_models_set['SVM'][1],
             'c-', label='SVM (AUC = %0.3f)' % auc(roc_models_set['SVM'][0], roc_models_set['SVM'][1]))
    plt.plot(roc_models_set['RF'][0], roc_models_set['RF'][1],
            'y-', label='RF (AUC = %0.3f)' % auc(roc_models_set['RF'][0], roc_models_set['RF'][1]))
    plt.plot(roc_models_set['NB'][0], roc_models_set['NB'][1],
            'm-', label='NB (AUC = %0.3f)' % auc(roc_models_set['NB'][0], roc_models_set['NB'][1]))
    plt.plot(roc_models_set['XGB'][0], roc_models_set['XGB'][1],
            'r-', label='XGBoost (AUC = %0.3f)' % auc(roc_models_set['XGB'][0], roc_models_set['XGB'][1]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels_new =[]
    for i in range(len(labels)):
        data_each = ''
        data_each = re.findall("[+-]?\d+\.\d+", labels[i])[0]
        labels_new.append(float(data_each))
    handles, labels = zip(*[(handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: labels_new[k], reverse=True)])
    plt.legend(handles, labels, loc='best')


    plt.subplot(1,2,2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(roc_models_set['XGB'][0], roc_models_set['XGB'][1],
            'r-', label='XGBoost (AUC = %0.3f)' % auc(roc_models_set['XGB'][0], roc_models_set['XGB'][1]))
    plt.plot(roc_scores_set['oasis'][0], roc_scores_set['oasis'][1],
            'b-', label='OASIS (AUC = %0.3f)' % auc(roc_scores_set['oasis'][0], roc_scores_set['oasis'][1]))
    plt.plot(roc_scores_set['apsiii'][0], roc_scores_set['apsiii'][1],
            'g-', label='APSIII (AUC = %0.3f)' % auc(roc_scores_set['apsiii'][0], roc_scores_set['apsiii'][1]))
    plt.plot(roc_scores_set['saps'][0], roc_scores_set['saps'][1],
            'c-', label='SAPS (AUC = %0.3f)' % auc(roc_scores_set['saps'][0], roc_scores_set['saps'][1]))
    plt.plot(roc_scores_set['sofa'][0], roc_scores_set['sofa'][1],
            'm-', label='SOFA (AUC = %0.3f)' % auc(roc_scores_set['sofa'][0], roc_scores_set['sofa'][1]))
    if train_test_name == 'MIMIC_eICU-eICU':
        plt.plot(roc_scores_set['apache_iv'][0], roc_scores_set['apache_iv'][1],
                'y-', label='APACHE-IV (AUC = %0.3f)' % auc(roc_scores_set['apache_iv'][0], roc_scores_set['apache_iv'][1]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels_new =[]
    for i in range(len(labels)):
        data_each = ''
        data_each = re.findall("[+-]?\d+\.\d+", labels[i])[0]
        labels_new.append(float(data_each))
    handles, labels = zip(*[(handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: labels_new[k], reverse=True)])
    plt.legend(handles, labels, loc='best')
    plt.savefig(path_save + 'roc_model_score_compare.png')
    plt.show()


def model_score_perf_matric_95ci(data_model, threshold_set_model, data_score, threshold_set_score, train_test_name, path_save):
    
    stats_bts_final, stats_final, data, threshold_set = pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame()
    data = pd.merge(data_model, data_score, on=['id','true_label'])
    threshold_set = threshold_set_model.append(threshold_set_score[threshold_set_model.columns.values.tolist()]).reset_index(drop=True)
    del data_model, threshold_set_model, data_score, threshold_set_score

    md_all = ['lr', 'svm', 'xgb', 'rf', 'nb', 'apsiii', 'sofa', 'oasis', 'saps']
    if train_test_name == 'MIMIC_eICU-eICU':
        md_all.insert(len(md_all), 'apache_iv')

    for md in md_all:
        data_each, stats_bts_each, stats_each = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        data_each = data[['true_label', md]]
        data_each.rename(columns={md: 'probability'}, inplace=True)
        stats_bts_each, stats_each = guef.model_performance_params_bootstrap_95CI(
                data=data_each, 
                data_ths=threshold_set.loc[threshold_set['name'] == md, 'threshold'].values[0],
                num_iterations=500
                )
        stats_bts_each['name'] = md
        stats_each = [md] + stats_each.values.tolist()[0]
        stats_final.append(stats_each)
        stats_bts_final = stats_bts_final.append(stats_bts_each)
    stats_final = pd.DataFrame(stats_final)
    stats_final.columns = ['name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    stats_final.to_csv(path_save + 'model_score_perf_matric_95ci.csv', index=False)
    stats_bts_final.to_csv(path_save + 'model_score_perf_matric_bts.csv', index=False)


def basic_info_merge(data):
    # get the basic info: age, gender, ethnicity, bmi
    data_basic_info = pd.DataFrame()
    data_basic_info_1 = pd.DataFrame()
    for db_name in ['mimic', 'eicu', 'ams']:
        data_each = pd.DataFrame()
        if db_name != 'ams':
            data_each = pd.read_csv('D:/project/older_score/data/' + db_name + '.csv')[['id', 'ethnicity']]
        else:
            data_each = pd.read_csv('D:/project/older_score/data/' + db_name + '.csv')['id'].to_frame()
            data_each['ethnicity'] = 'unknown'    
        data_basic_info_1 = data_basic_info_1.append(data_each)
    data_basic_info_1 = data_basic_info_1.reset_index(drop=True)
    data_basic_info_2 = pd.DataFrame()
    for db_name in ['MIMIC_eICU_use', 'eICU_extra', 'ams_use', 'MIMIC_temporal']:
        data_each = pd.DataFrame()
        data_each = pd.read_csv('D:/project/older_score/data/model_use/miceforest_lgb/' + db_name + '.csv')[['id', 'age', 'gender', 'bmi']]
        data_basic_info_2 = data_basic_info_2.append(data_each)
    data_basic_info_2 = data_basic_info_2.reset_index(drop=True)
    data_basic_info = pd.merge(data_basic_info_2, data_basic_info_1, on='id')
    del data_basic_info_1, data_basic_info_2
    # classify: age, gender, ethnicity - no need, bmi   
    data_basic_info['age_group'] = ''
    data_basic_info.loc[data_basic_info['age'] >= 80, 'age_group'] = 'old-old' 
    data_basic_info.loc[data_basic_info['age'] < 80, 'age_group'] = 'young-old'
    data_basic_info['bmi_group'] = ''
    data_basic_info.loc[data_basic_info['bmi'] >= 30, 'bmi_group'] = 'obesity' 
    data_basic_info.loc[((data_basic_info['bmi'] < 30) & (data_basic_info['bmi'] >= 25)), 'bmi_group'] = 'overweight'
    data_basic_info.loc[((data_basic_info['bmi'] < 25) & (data_basic_info['bmi'] >= 18.5)), 'bmi_group'] = 'normal'
    data_basic_info.loc[data_basic_info['bmi'] < 18.5, 'bmi_group'] = 'underweight'         
    data_basic_info['gender_group'] = ''
    data_basic_info['gender'] = data_basic_info['gender'].astype(int)
    data_basic_info.loc[data_basic_info['gender'] == 0, 'gender_group'] = 'female'
    data_basic_info.loc[data_basic_info['gender'] == 1, 'gender_group'] = 'male'
    data_basic_info['ethnicity_group'] = data_basic_info['ethnicity']
    # merge the basic info
    data = pd.merge(data, data_basic_info, how="left", on=["id"])
    del data_basic_info

    return data


def sub_model_score_perf_matric_95ci(data_model, threshold_set_model, data_score, threshold_set_score, train_test_name, path_save):
    
    data, threshold_set = pd.DataFrame(), pd.DataFrame()
    data = pd.merge(data_model, data_score, on=['id','true_label'])
    threshold_set = threshold_set_model.append(threshold_set_score[threshold_set_model.columns.values.tolist()]).reset_index(drop=True)
    del data_model, threshold_set_model, data_score, threshold_set_score

    data = basic_info_merge(data)
    sub_list = {}
    sub_list = {'age_group':['all', 'young-old','old-old'], \
                'gender_group':['all', 'female', 'male'], \
                'ethnicity_group':['all', 'asian', 'black', 'hispanic', 'white'], \
                'bmi_group':['all', 'underweight', 'normal', 'overweight', 'obesity']
                }
    if train_test_name == 'MIMIC_eICU-Ams':
        sub_list.pop('ethnicity_group', None)

    stats_bts_final, stats_final = pd.DataFrame(), []
    for sub_group in sub_list.keys():
        for sub_group_value in sub_list[sub_group]:
            md_all = ['lr', 'svm', 'xgb', 'rf', 'nb', 'apsiii', 'sofa', 'oasis', 'saps']
            if train_test_name == 'MIMIC_eICU-eICU':
                md_all.insert(len(md_all), 'apache_iv')
            for md in md_all:
                data_each, stats_new_each, stats_each = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                if sub_group_value == 'all':
                    data_each = data[['true_label', md]]
                else: 
                    data_each = data.loc[data[sub_group] == sub_group_value, ['true_label', md]].reset_index(drop=True)
                data_each.rename(columns={md: 'probability'}, inplace=True)
                stats_bts_each, stats_each = guef.model_performance_params_bootstrap_95CI(
                        data=data_each, 
                        data_ths=threshold_set.loc[threshold_set['name'] == md, 'threshold'].values[0],
                        num_iterations=500
                        )
                stats_each = [sub_group, sub_group_value, md] + stats_each.values.tolist()[0]
                stats_final.append(stats_each)
                stats_bts_each['sub_group'] = sub_group
                stats_bts_each['sub_group_value'] = sub_group_value
                stats_bts_each['name'] = md
                stats_bts_final = stats_bts_final.append(stats_bts_each)
    stats_final = pd.DataFrame(stats_final)
    stats_final.columns = ['sub_group_name', 'sub_group_value', 'name', 'roc_auc', 'sensitivity', \
                            'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
                            ]
    stats_final.to_csv(path_save + 'sub_model_score_perf_matric_95ci.csv', index=False)        
    stats_bts_final.to_csv(path_save + 'sub_model_score_perf_matric_bts.csv', index=False)


def part_fea_perf_matric_95ci(data, threshold_set, path_save):

    md_all = data.columns.values.tolist()
    md_all = [e for e in md_all if e not in ('id', 'true_label')]

    stats_bts_final, stats_final = pd.DataFrame(), []
    for md in md_all:
        data_each, stats_bts_each, stats_each = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        data_each = data[['true_label', md]]
        data_each.rename(columns={md: 'probability'}, inplace=True)
        stats_bts_each, stats_each = guef.model_performance_params_bootstrap_95CI(
                data=data_each, 
                data_ths=threshold_set.loc[threshold_set['name'] == md, 'threshold'].values[0],
                num_iterations=500
                )
        stats_bts_each['name'] = md
        stats_each = [md] + stats_each.values.tolist()[0]
        stats_final.append(stats_each)
        stats_bts_final = stats_bts_final.append(stats_bts_each)
    stats_final = pd.DataFrame(stats_final)
    stats_final.columns = ['name', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    stats_final.to_csv(path_save + 'part_fea_perf_matric_95ci.csv', index=False)
    stats_bts_final.to_csv(path_save + 'part_fea_perf_matric_bts.csv', index=False)    


# matlab needed - auc and ap
def generate_matplab_plot_95ci_split(data):
    
    data_use = pd.DataFrame()
    data_use = data[['name', 'roc_auc', 'ap']]

    data_use[['auc', 'auc_range']] = data_use['roc_auc'].str.split(' ', 1, expand=True)
    data_use[['auc_lower', 'auc_upper']] = data_use['auc_range'].str.split('-', 1, expand=True)
    data_use[['auc_lower', 'auc_upper']] = data_use[['auc_lower', 'auc_upper']].replace('\(|\)','',regex=True)

    data_use[['ap', 'ap_range']] = data_use['ap'].str.split(' ', 1, expand=True)    
    data_use[['ap_lower', 'ap_upper']] = data_use['ap_range'].str.split('-', 1, expand=True)
    data_use[['ap_lower', 'ap_upper']] = data_use[['ap_lower', 'ap_upper']].replace('\(|\)','',regex=True)

    data_use = data_use[['name', 'auc', 'auc_lower', 'auc_upper', 'ap', 'ap_lower', 'ap_upper']]
    data_use[['auc', 'auc_lower', 'auc_upper', 'ap', 'ap_lower', 'ap_upper']] = \
            data_use[['auc', 'auc_lower', 'auc_upper', 'ap', 'ap_lower', 'ap_upper']].astype(float)
    data_use['auc_lower'] = data_use['auc'] - data_use['auc_lower']
    data_use['auc_upper'] = data_use['auc_upper'] - data_use['auc']
    data_use['ap_lower'] = data_use['ap'] - data_use['ap_lower']
    data_use['ap_upper'] = data_use['ap_upper'] - data_use['ap']

    # rename, re-rank for matlab easily use, and drop svm model
    data_use.rename(columns={'name': 'model_name'}, inplace=True)
    ml_list = ['xgb', 'rf', 'svm', 'lr', 'nb']
    data_use_part1, data_use_part2 = pd.DataFrame(), pd.DataFrame()
    data_use_part1 = data_use[data_use['model_name'].isin(ml_list)]
    data_use_part1 = data_use_part1.sort_values(['auc'], ascending=False)
    data_use_part2 = data_use[~data_use['model_name'].isin(ml_list)]
    data_use_part2 = data_use_part2.sort_values(['auc'], ascending=False)
    data_use = data_use_part1.append(data_use_part2).reset_index(drop=True)
    # data_use = data_use[~(data_use['model_name'] == 'svm')]

    return data_use


# change the results to tableone format to directly calculate the p values (models & scores)
def auc_ap_model_score_vs_pvalue(data, test_set_name, path_save):
    for pvalue_name in ['roc_auc', 'ap']:
        data_pvalue_use = []
        model_name_list = ['lr', 'rf', 'nb', 'sofa', 'apsiii', 'saps', 'oasis']
        if test_set_name == 'eICU':
            model_name_list.insert(len(model_name_list), 'apache_iv')
        for name in model_name_list:
            data_pvalue = pd.DataFrame()
            data_pvalue = data.loc[data['name'] == name, pvalue_name].values.reshape(-1,1)
            if len(data_pvalue_use) == 0:
                data_pvalue_use = data_pvalue
            else:
                data_pvalue_use = np.hstack((data_pvalue_use, data_pvalue))
        data_pvalue_use = pd.DataFrame(data_pvalue_use, columns=model_name_list)
        data_pvalue_use['label'] = 0
        # append xgb results
        data_pvalue_xgb = pd.DataFrame()
        data_pvalue_xgb = data.loc[data['name'] == 'xgb', pvalue_name].reset_index(drop=True)
        data_pvalue_xgb = pd.concat([data_pvalue_xgb] * (len(model_name_list)), axis=1, ignore_index=True)
        data_pvalue_xgb.columns = model_name_list
        data_pvalue_xgb['label'] = 1
        data_pvalue_use = data_pvalue_use.append(data_pvalue_xgb).reset_index(drop=True)
        groupby, columns = '', []
        groupby = 'label'
        columns = data_pvalue_use.columns.values.tolist()
        nonnormal = [x for x in columns if x != 'label']
        categorical = []
        group_table = TableOne(data_pvalue_use, columns=columns, categorical=categorical, \
                               groupby=groupby, nonnormal=nonnormal, decimals=3, pval=True)
        group_table.to_excel(path_save + pvalue_name + '_pvalue.xlsx')


# matlab needed - subgroup analysis of bar plot using AUC
def sub_auc_xgb_plot_95ci_split(data):

    data_final = pd.DataFrame() # save the output results
    data_final = data.loc[data['name'] == 'xgb', ['sub_group_name', 'sub_group_value', 'roc_auc']].reset_index(drop=True)
    data_final[['auc', 'auc_range']] = data_final['roc_auc'].str.split(' ', 1, expand=True)
    data_final[['auc_lower', 'auc_upper']] = data_final['auc_range'].str.split('-', 1, expand=True)
    data_final[['auc_lower', 'auc_upper']] = data_final[['auc_lower', 'auc_upper']].replace('\(|\)','',regex=True)
    data_final[['auc', 'auc_lower', 'auc_upper']] = data_final[['auc', 'auc_lower', 'auc_upper']].astype(float)
    data_final['auc_lower'] = data_final['auc'] - data_final['auc_lower']
    data_final['auc_upper'] = data_final['auc_upper'] - data_final['auc']
    data_final = data_final.rename(columns={'auc': 'auc_median'})
    data_final.drop(['auc_range'], axis=1, inplace=True)
    # save age_group all type, drop other all types
    data_final_part1, data_final_part2 = pd.DataFrame(), pd.DataFrame()
    data_final_part1 = data_final.loc[data_final['sub_group_name'] == 'age_group']
    data_final_part2 = data_final.loc[~(data_final['sub_group_name'] == 'age_group')]
    data_final_part2 = data_final_part2.loc[~(data_final['sub_group_value'] == 'all')]
    data_final = data_final_part1.append(data_final_part2).reset_index(drop=True)

    return data_final


# matlab part of features' performance plot
def part_fea_auc_xgb_plot_95ci_split(data):

    data_final = pd.DataFrame() # save the output results
    data_final = data.copy()
    data_final[['auc', 'auc_range']] = data_final['roc_auc'].str.split(' ', 1, expand=True)
    data_final[['auc_lower', 'auc_upper']] = data_final['auc_range'].str.split('-', 1, expand=True)
    data_final[['auc_lower', 'auc_upper']] = data_final[['auc_lower', 'auc_upper']].replace('\(|\)','',regex=True)
    data_final[['auc', 'auc_lower', 'auc_upper']] = data_final[['auc', 'auc_lower', 'auc_upper']].astype(float)
    data_final['auc_lower'] = data_final['auc'] - data_final['auc_lower']
    data_final['auc_upper'] = data_final['auc_upper'] - data_final['auc']
    data_final = data_final.rename(columns={'auc': 'auc_median'})
    data_final.drop(['auc_range'], axis=1, inplace=True)
    data_final['fea_num'] = data_final['name'].str.replace('xgb_', '').astype(int)
    data_final.rename(columns={'name': 'model_name'}, inplace=True)
    
    return data_final[['model_name', 'auc_median', 'auc_lower', 'auc_upper', 'fea_num']]
    			

# matlab drop features of forest plot data generate
def drop_fea_auc_xgb_plot_95ci_split(data):
    # https://devenum.com/write-dictionary-to-text-file-in-python/
    # https://www.mathworks.com/matlabcentral/fileexchange/71020-forest-plot-for-visualisation-of-multiple-odds-ratios
    data_final = pd.DataFrame() # save the output results
    data_final = data.copy()
    data_final[['diff_auc', 'diff_auc_range']] = data_final['diff_roc_auc'].str.split(' ', 1, expand=True)
    data_final[['diff_auc_lower', 'diff_auc_upper']] = data_final['diff_auc_range'].str.split(' - ', 1, expand=True)
    data_final[['diff_auc_lower', 'diff_auc_upper']] = data_final[['diff_auc_lower', 'diff_auc_upper']].replace('\(|\)','',regex=True)
    data_final[['diff_auc', 'diff_auc_lower', 'diff_auc_upper']] = data_final[['diff_auc', 'diff_auc_lower', 'diff_auc_upper']].astype(float)
    # due to the blobbogram.m meeting, no need to get the diff value
    # data_final['diff_auc_lower'] = data_final['diff_auc'] - data_final['diff_auc_lower']
    # data_final['diff_auc_upper'] = data_final['diff_auc_upper'] - data_final['diff_auc']
    data_final = data_final.rename(columns={'diff_auc': 'diff_auc_median'})
    data_final.drop(['diff_auc_range'], axis=1, inplace=True)
    data_final['feature_name'] = data_final['name'].str.replace('xgb_no_', '')
    # data_final.to_csv(path + 'drop_fea_models_compare_diff_auc_all_95CI_plot.csv', index=False)
    data_final.drop(['name'], axis=1, inplace=True)

    return data_final


# development, external, temporal set of tableone info
def dev_ext_temp_tableone(data_load_path, path_save):
    
    # 1. get the needed data and each part of patients
    MIMIC = pd.read_csv(data_load_path + 'mimic.csv').drop(['subject_id', 'hadm_id'], axis=1)
    eICU = pd.read_csv(data_load_path + 'eicu.csv').drop(['uniquepid', 'patienthealthsystemstayid'], axis=1)
    ams = pd.read_csv(data_load_path + 'ams.csv')
    MIMIC_eICU_use = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'MIMIC_eICU_use.csv')
    MIMIC_temporal = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'MIMIC_temporal.csv')
    eICU_extra = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'eICU_extra.csv')
    ams_use = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'ams_use.csv')
    development_id, external_1_id, external_2_id, temporal_id = [], [], [], []
    development_id = list(MIMIC_eICU_use['id'])
    external_1_id = list(eICU_extra['id'])
    external_2_id = list(ams_use['id'])
    temporal_id = list(MIMIC_temporal['id'])

    development_non_imputation, external_1_non_imputation, external_2_non_imputation, temporal_non_imputation = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    development_imputation, external_1_imputation, external_2_imputation, temporal_imputation = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    development_non_imputation = MIMIC.append(eICU[MIMIC.columns.values.tolist()]).reset_index(drop=True)
    development_non_imputation = development_non_imputation.loc[development_non_imputation['id'].isin(development_id)].reset_index(drop=True)
    development_non_imputation.loc[development_non_imputation['anchor_year_group'] == 2015, 'anchor_year_group'] = '2014 - 2016'
    development_non_imputation.loc[development_non_imputation['anchor_year_group'] == 2014, 'anchor_year_group'] = '2014 - 2016'
    external_1_non_imputation = eICU.loc[eICU['id'].isin(external_1_id)].reset_index(drop=True)
    ams = gudp.ams_preprocess(ams)
    external_2_non_imputation = ams.drop(['cci_score', 'code_status', 'code_status_eva_flag', 'pre_icu_los_day'], axis=1)
    temporal_non_imputation = MIMIC.loc[MIMIC['id'].isin(temporal_id)].reset_index(drop=True)

    development_imputation = MIMIC_eICU_use
    external_1_imputation = eICU_extra
    external_2_imputation = ams_use.drop(['code_status', 'code_status_eva_flag', 'pre_icu_los_day'], axis=1)
    temporal_imputation = MIMIC_temporal

    del MIMIC, eICU, ams, MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal

    # 2. get categorical column's name
    categorical_all = ['activity_bed', 'activity_eva_flag', 'activity_sit', 'activity_stand'
    , 'admission_type', 'agegroup', 'anchor_year_group', 'epinephrine'
    , 'code_status', 'code_status_eva_flag', 'death_hosp', 'delirium_eva_flag'
    , 'delirium_flag', 'dobutamine', 'dopamine', 'electivesurgery', 'ethnicity'
    , 'fall_risk', 'fall_risk_eva_flag', 'first_careunit', 'gender', 'norepinephrine'
    , 'region', 'teachingstatus', 'vent', 'weightgroup', 'heightgroup']
    categorical_all = categorical_all + [s for s in development_imputation.columns.values.tolist() if '_flag' in s]
    categorical_all = list(set(categorical_all))


    # 3. get the non-imputation data tableone
    non_imputation_set = {}
    non_imputation_set = {'development_non_imputation': development_non_imputation, 'external_1_non_imputation': external_1_non_imputation, \
    'external_2_non_imputation': external_2_non_imputation, 'temporal_non_imputation': temporal_non_imputation
    }
    for i in non_imputation_set.keys():
        print(i)
        overall_table, group_table = [], []
        overall_table, group_table = gudp.cal_tableone_info(non_imputation_set[i], 'death_hosp', categorical_all)
        overall_table.to_excel(path_save + 'overall_' + i + '.xlsx')
        group_table.to_excel(path_save + 'group_' + i + '.xlsx')

    # 4. get the imputation data tableone
    imputation_set = {}
    imputation_set = {'development_imputation': development_imputation, 'external_1_imputation': external_1_imputation, \
    'external_2_imputation': external_2_imputation, 'temporal_imputation': temporal_imputation
    }
    for i in imputation_set.keys():
        print(i)
        overall_table, group_table = [], []
        overall_table, group_table = gudp.cal_tableone_info(imputation_set[i], 'death_hosp', categorical_all)
        overall_table.to_excel(path_save + 'overall_' + i + '.xlsx')
        group_table.to_excel(path_save + 'group_' + i + '.xlsx')


# development, external, temporal set of missing ratio info
def dev_ext_temp_missing_ratio(data_load_path, path_save):
    
    # 1. get the needed data and each part of patients
    MIMIC = pd.read_csv(data_load_path + 'mimic.csv').drop(['subject_id', 'hadm_id'], axis=1)
    eICU = pd.read_csv(data_load_path + 'eicu.csv').drop(['uniquepid', 'patienthealthsystemstayid'], axis=1)
    ams = pd.read_csv(data_load_path + 'ams.csv')
    MIMIC_eICU_use = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'MIMIC_eICU_use.csv')
    MIMIC_temporal = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'MIMIC_temporal.csv')
    eICU_extra = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'eICU_extra.csv')
    ams_use = pd.read_csv(data_load_path + 'model_use/miceforest_lgb/' + 'ams_use.csv')
    development_id, external_1_id, external_2_id, temporal_id = [], [], [], []
    development_id = list(MIMIC_eICU_use['id'])
    external_1_id = list(eICU_extra['id'])
    external_2_id = list(ams_use['id'])
    temporal_id = list(MIMIC_temporal['id'])

    development_non_imputation, external_1_non_imputation, external_2_non_imputation, temporal_non_imputation = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    development_imputation, external_1_imputation, external_2_imputation, temporal_imputation = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    development_non_imputation = MIMIC.append(eICU[MIMIC.columns.values.tolist()]).reset_index(drop=True)
    development_non_imputation = development_non_imputation.loc[development_non_imputation['id'].isin(development_id)].reset_index(drop=True)
    development_non_imputation.loc[development_non_imputation['anchor_year_group'] == 2015, 'anchor_year_group'] = '2014 - 2016'
    development_non_imputation.loc[development_non_imputation['anchor_year_group'] == 2014, 'anchor_year_group'] = '2014 - 2016'
    external_1_non_imputation = eICU.loc[eICU['id'].isin(external_1_id)].reset_index(drop=True)
    external_2_non_imputation = ams
    temporal_non_imputation = MIMIC.loc[MIMIC['id'].isin(temporal_id)].reset_index(drop=True)

    gudp.calculate_missing_ratio(development_non_imputation).to_csv(path_save + 'development_missing.csv', index=False)
    gudp.calculate_missing_ratio(external_1_non_imputation).to_csv(path_save + 'external_1_missing.csv', index=False)
    gudp.calculate_missing_ratio(external_2_non_imputation).to_csv(path_save + 'external_2_missing.csv', index=False)
    gudp.calculate_missing_ratio(temporal_non_imputation).to_csv(path_save + 'temporal_missing.csv', index=False)


def feature_ranking_round(data_load_path, path_save):
    data_final = pd.DataFrame() 
    for i in ['all', 'young-old', 'old-old']:
        data_use = pd.DataFrame()
        data_use = pd.read_csv(data_load_path + i + '_features_ranking.csv')
        data_use['feature_importance_vals'] = data_use['feature_importance_vals'].round(3)
        data_use['feature_group'] = i
        data_final = data_final.append(data_use)
    data_final.to_csv(path_save + 'features_ranking_round.csv', index=False)


# https://medium.com/analytics-vidhya/how-probability-calibration-works-a4ba3f73fd4d
def older_calibration_curves_set(data_load_path, train_test_name, path_save):
    label_set = {}
    label_set = {'xgb': 'XGBoost', 'rf': 'Random Forest', 'lr': 'Logistics Regression', 'nb': 'Naive Bayes', \
        'apsiii': 'APSIII', 'sofa': 'SOFA', 'oasis': 'OASIS', 'saps': 'SAPS', 'apache_iv': 'APACHE IV'}

    for plot_type in ['models', 'scores']:
        data_need = pd.DataFrame()
        if plot_type == 'models':
            data_need = pd.read_csv(data_load_path + train_test_name + '/' + plot_type + '_pred_true.csv')
            data_need = data_need[['id', 'true_label', 'lr', 'nb', 'rf', 'xgb']]
        else:
            data_need = pd.read_csv(data_load_path + train_test_name + '/' + plot_type + '_pred_true.csv')
            data_extra = pd.DataFrame()
            data_extra = pd.read_csv(data_load_path + train_test_name + '/' + 'models_pred_true.csv')[['id', 'xgb']]
            data_need = pd.merge(data_need, data_extra, on=['id'])
            del data_extra

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot([0,1], [0,1], "b--", label="Perfectly calibrated")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Mean predicted value")
        # ax.set_title('Calibration plot (reliability curve)')

        plot_curves_set = []
        plot_curves_set = data_need.columns.to_list()
        plot_curves_set = [ele for ele in plot_curves_set if ele not in ['id', 'true_label']]
        for i in plot_curves_set:
            fraction_of_positives_n, mean_predicted_value_n = calibration_curve(data_need['true_label'], data_need[i], n_bins=10)
            ax.plot(mean_predicted_value_n, fraction_of_positives_n, "s-", label="%s" % (label_set[i]))

        ax.legend(loc="lower right")
        # plt.show()
        fig.savefig(path_save + train_test_name + '/plot_table/' + plot_type + '_calibration_curves.png', tight=True, quality=500)
        plt.close()


def sub_xgb_gossis_sms_perf_matric_95ci(data, train_test_name, path_save):
    # basic set
    models_ls, groups_ls = [], []
    models_ls = ['xgb', 'sms_prob']
    if train_test_name == 'MIMIC_eICU-eICU':
        models_ls = models_ls + ['gossis']
    groups_ls = ['age_group', 'gender_group', 'ethnicity']
    if train_test_name == 'MIMIC_eICU-Ams':
        groups_ls.remove('ethnicity')

    # result save
    with open(path_save + train_test_name.replace('MIMIC_eICU-', '') + '_xgb_gossis_sms_95ci.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model', 'group', 'group_cat', 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'])

    # acquire performance    
    for i in models_ls:
        for j in groups_ls:
            data_need = pd.DataFrame()
            data_need = data[['true_label', i, j]]
            parameters = {} 
            parameters, _ = guef.model_performance_params(data_need['true_label'], data_need[i], 'False', 0)
            groups_ls_cat = []
            if j == 'age_group':
                groups_ls_cat = ['all', 'young-old','old-old']
            elif j == 'gender_group':
                groups_ls_cat = ['all', 'female', 'male']
            else:
                groups_ls_cat = ['all', 'asian', 'black', 'hispanic', 'white']

            for g in groups_ls_cat:
                data_each = pd.DataFrame()
                if g == 'all':
                    data_each = data_need
                else:
                    data_each = data_need.loc[data_need[j] == g].reset_index(drop=True)
                data_each.rename(columns={i: 'probability'}, inplace=True)
                data_each = data_each[['true_label', 'probability']]
                stats_result_each = []
                _, stats_result_each = guef.model_performance_params_bootstrap_95CI(data_each, parameters['threshold'], 500)
                stats_result_each = [i, j, g] + list(stats_result_each.values[0])
                result_all = []
                result_all = open(path_save + train_test_name.replace('MIMIC_eICU-', '') + '_xgb_gossis_sms_95ci.csv', 'a', newline='')
                writer = csv.writer(result_all)
                writer.writerow(stats_result_each)
                result_all.close()    




