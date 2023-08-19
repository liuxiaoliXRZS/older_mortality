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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import csv
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
import os
import miceforest as mf


# define the outliers' drop function and remove values out of physiological reasonable range
def outlier_value_nan(data, outlier_range_check_new):
    """
    :param data: dataframe - the input data generating from the databases ['id', 'f1', 'f2']
    :param outlier_range_check_new: dataframe - the upper and lower bound of features [f_name, lower bound, upper bound]
    :return data: dataframe - remove the out of range values
    """
    # change columns' name to lowercase (in order to map the outlier_range_check's name)
    columns_name = []
    columns_name = [x.lower() for x in data.columns.values.tolist()]
    data.columns = columns_name
    for i in range(outlier_range_check_new.shape[0]):
        if outlier_range_check_new['index_name'].tolist()[i].lower() in data.columns.values.tolist():
            data.loc[
                (data[outlier_range_check_new['index_name'].tolist()[i].lower()] > outlier_range_check_new.loc[
                    i].upper_bound) |
                (data[outlier_range_check_new['index_name'].tolist()[i].lower()] < outlier_range_check_new.loc[
                    i].lower_bound),
                outlier_range_check_new['index_name'].tolist()[i].lower()] = np.nan  # and : & ; or : |
    return data


# calculate missing ratio
def calculate_missing_ratio(data):
    """
    :param data: dataframe ['id', 'f1', 'f2']
    :return missing_value_df: ['fea_name', 'missing_ratio']
    """
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns, 'percent_missing': percent_missing})
    missing_value_df['percent_missing'] = missing_value_df['percent_missing'].round(2)
    return missing_value_df


# table one info
def cal_tableone_info(data, group_name, categorical_all):
    """
    :param data: the import data used to calculate tableone information
    :para group_name: the group name like 'death_hosp'
    :param categorical_all: all categorical names of four databases
    :return overall_table: without considering the group label
    :return groupby_table: considering the group label
    """
    columns, categorical, groupby, nonnormal = [], [], [], []
    columns = data.columns.values.tolist()
    columns.remove('id')  # change to the needing columns' name
    categorical = list(set(data.columns.values.tolist()).intersection(set(categorical_all)))
    nonnormal = [x for x in columns if x != group_name]
    groupby = group_name
    overall_table = TableOne(data, columns, categorical, nonnormal=columns)
    group_table = TableOne(data, columns, categorical, groupby, nonnormal, pval=True)
    return overall_table, group_table


# string to int type
def older_string_int(data):
    """
    :param data: the import data containing character value's columns
    :return data: character value maps to int value
    """
    string_column_name = {'ethnicity':{'asian':0, 'white':3, 'hispanic':2, 'black':1, 'other':4, 'unknown':5},
                          'gender':{'F':0, 'M':1},
                          'admission_type':{'ELECTIVE':0, 'Floor':0, 'Intermediate care unit':0,
                                            'Other/Unknown':0, 'EMERGENCY':1, 'URGENT':1,
                                            'OBSERVATION': 0, 'planned': 0, 'unplanned':1}
                          }

    for i in string_column_name:
        data[i] = data[i].map(string_column_name[i])

    return data


# special preprocess ams data before similar processing as mimic and eicu: get age, weight, height
def ams_preprocess(data):
    # data[['agegroup', 'weightgroup', 'heightgroup', 'ethnicity']]
    np.random.seed(9001)
    data['age'], data['weight'], data['height'], data[
        'bmi'] = pd.Series(), pd.Series(), pd.Series(), pd.Series()
        
    # age - agegroup
    data['age'].loc[data['agegroup'] == '70-79'] = \
        np.random.randint(70, 79, size=(1, data['agegroup'].value_counts()['70-79']))[0]
    data['age'].loc[data['agegroup'] == '80+'] = \
        np.random.randint(80, 100, size=(1, data['agegroup'].value_counts()['80+']))[0]

    # weight - weightgroup
    data['weight'].loc[data['weightgroup'] == '59-'] = \
        np.random.randint(340, 599, size=(1, data['weightgroup'].value_counts()['59-']))[0] / 10
    data['weight'].loc[data['weightgroup'] == '60-69'] = \
        np.random.randint(600, 699, size=(1, data['weightgroup'].value_counts()['60-69']))[0] / 10
    data['weight'].loc[data['weightgroup'] == '70-79'] = \
        np.random.randint(700, 799, size=(1, data['weightgroup'].value_counts()['70-79']))[0] / 10
    data['weight'].loc[data['weightgroup'] == '80-89'] = \
        np.random.randint(800, 899, size=(1, data['weightgroup'].value_counts()['80-89']))[0] / 10
    data['weight'].loc[data['weightgroup'] == '90-99'] = \
        np.random.randint(900, 999, size=(1, data['weightgroup'].value_counts()['90-99']))[0] / 10
    data['weight'].loc[data['weightgroup'] == '100-109'] = \
        np.random.randint(1000, 1099, size=(1, data['weightgroup'].value_counts()['100-109']))[0] / 10
    data['weight'].loc[data['weightgroup'] == '110+'] = \
        np.random.randint(1100, 1750, size=(1, data['weightgroup'].value_counts()['110+']))[0] / 10

    # height - heightgroup
    data['height'].loc[data['heightgroup'] == '159-'] = \
        np.random.randint(1400, 1599, size=(1, data['heightgroup'].value_counts()['159-']))[0] / 10
    data['height'].loc[data['heightgroup'] == '160-169'] = \
        np.random.randint(1600, 1699, size=(1, data['heightgroup'].value_counts()['160-169']))[0] / 10
    data['height'].loc[data['heightgroup'] == '170-179'] = \
        np.random.randint(1700, 1799, size=(1, data['heightgroup'].value_counts()['170-179']))[0] / 10
    data['height'].loc[data['heightgroup'] == '180-189'] = \
        np.random.randint(1800, 1899, size=(1, data['heightgroup'].value_counts()['180-189']))[0] / 10
    data['height'].loc[data['heightgroup'] == '190+'] = \
        np.random.randint(1900, 2058, size=(1, data['heightgroup'].value_counts()['190+']))[0] / 10

    data['ethnicity'] = 'unknown'
    
    return data


# data process - part1
def data_process1_older_score(data_use, data_use_name, outlier_range_check, result_path):
    """
    :param data_use: initial data without processing
    :para data_use_name: the group name like 'death_hosp'
    :param outlier_range_check: the initial name outlier like hearrate, resp_rate
    :param result_path: save path of results
    :return data_use: remove outlier
    """    

    data_use_final = pd.DataFrame() # create to generate

    # [0]. special preprocess ams
    if data_use_name == 'ams':
        data_use = ams_preprocess(data_use)

    # [1]. drop outliers - Amplification the expression: add feature_max/min/mean
    outlier_range_check_new = pd.DataFrame()
    outlier_range_check_new = pd.concat([outlier_range_check, outlier_range_check, outlier_range_check, outlier_range_check], axis=0)
    outlier_range_check_new.rename(columns={'Unnamed: 0': 'index_name'}, inplace=True)
    outlier_range_check.rename(columns={'Unnamed: 0': 'index_name'}, inplace=True)
    # according to the statistic features' name to generate the outlier check columns
    add_fea = ['', '_min', '_max', '_mean']
    new_index = []
    for i in add_fea:
        for j in outlier_range_check['index_name'].tolist():
            new_index.append(j + i)
    outlier_range_check_new['index_name'] = new_index
    outlier_range_check_new.reset_index(drop=True, inplace=True)

    data_use = outlier_value_nan(data_use, outlier_range_check_new)

    # [2]. calculate missing ratio
    calculate_missing_ratio(data_use).to_csv(result_path + data_use_name + '_missing.csv', index=False)

    # get the impute data's tableone info to support to check data
    categorical_all = ['activity_bed', 'activity_eva_flag', 'activity_sit', 'activity_stand'
        , 'admission_type', 'agegroup', 'anchor_year_group', 'epinephrine'
        , 'code_status', 'code_status_eva_flag', 'death_hosp'
        , 'dobutamine', 'dopamine', 'electivesurgery', 'ethnicity'
        , 'first_careunit', 'gender', 'norepinephrine'
        , 'region', 'teachingstatus', 'vent', 'weightgroup', 'heightgroup']
    
    if data_use_name == 'ams':
        overall_table, group_table = cal_tableone_info(data_use.drop(['patientid', 
            'code_status', 'code_status_eva_flag', 'pre_icu_los_day', 'cci_score'], axis=1), 'death_hosp', categorical_all)
    else:
        overall_table, group_table = cal_tableone_info(
            data_use.drop(['uniquepid', 'patienthealthsystemstayid', 'subject_id', 'hadm_id'], errors='ignore', axis=1), 'death_hosp', categorical_all)
    overall_table.to_excel(result_path + 'overall_' + data_use_name + '.xlsx')
    group_table.to_excel(result_path + 'group_' + data_use_name + '.xlsx')
    del overall_table, group_table

    # [3]. map string to int
    data_use = older_string_int(data_use)

    # [4] remove no need variables
    drop_names = ['agegroup', 'anchor_year_group', 'apache_iva', 'apache_iva_prob', 'deathtime_icu_hour',
        'first_careunit', 'hadm_id', 'heightgroup', 'hospitalid', 'los_hospital_day', 'los_icu_day', 'patienthealthsystemstayid',
        'patientid', 'predictedhospitallos_iv', 'predictedhospitallos_iva', 'predictediculos_iv', 'predictediculos_iva',
        'region', 'subject_id', 'teachingstatus', 'uniquepid', 'weightgroup', 'bmi',
        'troponin_max', 'fibrinogen_min', 'bnp_max', 'apache_iv', 'apache_iv_prob',
        'oasis', 'oasis_prob', 'saps', 'saps_prob', 'sofa', 'sofa_prob', 'apsiii', 'apsiii_prob'
        ]
    data_use.drop(drop_names, axis=1, inplace=True, errors='ignore')

    return data_use


# generate different source of data before imputation
def get_data_diff_source(MIMIC_initial, eICU_initial, ams_initial):
    """
    :param MIMIC_initial: mimic data (drop outlier)
    :param eICU_initial: eicu data (drop outlier)
    :param ams_initial: ams data (drop outlier)
    :return MIMIC_use: 2001-2016
    :return MIMIC_temporal: 2017-2019
    :return eICU_use: 13 hps [264, 73, 420, 243, 122, 188, 281, 148, 443, 167, 176, 338, 283]
    :return eICU_extra: other hps
    :return ams_use: ams
    :return MIMIC_eICU_use: merge MIMIC_use and eICU_use    
    """ 
    # get special used ids
    # here we need to upload the raw data to get the hospitals and admission year info
    # we merged 13 hps to the development set
    path = 'D:/project/older_score/data/' # make sure the same as data_initial_path in the 'older_model_main.py'
    mimic_id, eicu_id = pd.DataFrame(), pd.DataFrame()
    mimic_id = pd.read_csv(path + 'mimic.csv')
    mimic_id = mimic_id.loc[mimic_id['anchor_year_group'] == '2017 - 2019', 'id'].values.tolist()
    eicu_id = pd.read_csv(path + 'eicu.csv')
    eicu_id = eicu_id[eicu_id['hospitalid'].isin([264, 73, 420, 243, 122, 188, 281, 148, 443, 167, 176, 338, 283])].id.values.tolist()
    # get different usage datasets
    MIMIC_use, MIMIC_temporal, eICU_use, eICU_extra, MIMIC_eICU_use = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    MIMIC_use = MIMIC_initial[~MIMIC_initial['id'].isin(mimic_id)].reset_index(drop=True)
    MIMIC_temporal = MIMIC_initial[MIMIC_initial['id'].isin(mimic_id)].reset_index(drop=True)
    eICU_use = eICU_initial[eICU_initial['id'].isin(eicu_id)].reset_index(drop=True)
    eICU_extra = eICU_initial[~eICU_initial['id'].isin(eicu_id)].reset_index(drop=True)
    ams_use = ams_initial # without processing
    MIMIC_use = MIMIC_use.reindex(sorted(MIMIC_use.columns), axis=1)
    eICU_use = eICU_use.reindex(sorted(eICU_use.columns), axis=1)
    MIMIC_eICU_use = MIMIC_use.append(eICU_use) # merge MIMIC_use and eICU_use
    MIMIC_eICU_use.reset_index(drop=True, inplace=True)
    eICU_extra.reset_index(drop=True, inplace=True)


    return MIMIC_eICU_use, MIMIC_temporal, eICU_extra, ams_use


# add flag: indicate the measurement existing or not
def add_flag(data, flag_columns):
    """
    :param data: the import data containing columns needed to add the corresponding columns
    :param flag_columns: list of all flag features' name
    :return data_new: add the flag columns with zero or one value to indicate no-record or having record
    """    
    # flag_columns = ['pao2fio2ratio_vent', 'pao2fio2ratio_novent', 'bilirubin_max'
    #                 , 'albumin_min', 'alp_max', 'alt_max', 'ast_max', 'baseexcess_min', 'fio2_max'
    #                 , 'lactate_max', 'lymphocytes_max', 'lymphocytes_min', 'magnesium_max', 'neutrophils_min'
    #                 , 'paco2_max', 'pao2_min', 'ptt_max']
    data_flag = data[flag_columns]
    data_flag = data_flag.notnull().astype('int')
    flag_columns_new = []
    for i in data_flag.columns.tolist():
        flag_columns_new.append(i + '_flag')
    data_flag.columns = flag_columns_new
    data_new = pd.DataFrame()
    data_new = pd.concat([data, data_flag], axis=1)
    return data_new


# imputation - add indicators (0 [non existing] or 1 [existing])
def imputation_median_indicators(data, features_name, features_name_flag, data_name, data_save_path):
    """
    :param data: the import data containing empty values
    :param features_name:
    :param features_name_flag:
    :param data_name:
    :return data_new: the imputed dataframe
    """

    # data_part1: median value for missing
    # data_part2: 0 for missing
    data_part1, data_part2 = pd.DataFrame(), pd.DataFrame()
    names_part1 = [] # no flag features name
    names_part1 = list(set(data.columns.tolist()).difference(set(features_name + features_name_flag)))
    data_part1 = data[names_part1]
    data_part1_new = pd.DataFrame()
    names_part1_new = list(set(names_part1).difference(['id']))
    if data_name == 'MIMIC_eICU_use':
        # only developing set could be used as the imputation reference dataset
        # split to development, calibration, internal validation datasets
        data_part1_dev = pd.DataFrame()
        data_part1_dev, _ = train_test_split(data_part1, test_size=0.2, random_state=0, shuffle=True)
        data_part1_median = data_part1_dev[names_part1_new].median()
        del data_part1_dev       
        data_part1_new = data_part1[names_part1_new].fillna(data_part1_median)
        data_part1_new = pd.concat([data[['id']], data_part1_new], axis=1)
        data_part1_new.columns = ['id'] + names_part1_new
        data_part1_median.to_csv(data_save_path + 'MIMIC_eICU_median.csv')
    else:
        data_part1_median = pd.read_csv(data_save_path + 'MIMIC_eICU_median.csv')
        data_part1_median.rename(columns={'Unnamed: 0': 'fea_name', '0': 'value'}, inplace=True)
        data_part1_median = dict(data_part1_median.values)
        data_part1_new = data_part1[names_part1_new].fillna(data_part1_median)
        data_part1_new = pd.concat([data[['id']], data_part1_new], axis=1)
        data_part1_new.columns = ['id'] + names_part1_new

    data_part2 = data[features_name]
    data_part2.fillna(0, inplace=True)
    data_part2_new = pd.DataFrame()
    data_part2_new = pd.concat([data[features_name_flag], data_part2], axis=1)

    del data_part1, data_part2

    data_imputation = pd.DataFrame()
    data_imputation = pd.concat([data_part1_new, data_part2_new], axis=1)

    del data_part1_new, data_part2_new

    return data_imputation


# impute values (features with flag: will impute zero; features without flag: will impute median value)
def generate_data_imputation_median_indicator(data, data_name, data_save_path):
    """
    :param data: the import data containing empty values
    :param data_name: data name to indicate use the generated median values or obtain median values
    :param data_save_path: imputed data save path
    :return data_use_final: directly save
    """    
    features_name = ['pao2fio2ratio_vent', 'pao2fio2ratio_novent', 'bilirubin_max'
        , 'albumin_min', 'alp_max', 'alt_max', 'ast_max', 'baseexcess_min', 'fio2_max'
        , 'lactate_max', 'lymphocytes_max', 'lymphocytes_min', 'neutrophils_min'
        , 'paco2_max', 'pao2_min']

    # [1] add flag with value
    data_use_final = pd.DataFrame()
    data_use_final = add_flag(data, features_name)

    # [2] imputation with 0 or median values 
    features_name_flag = [s + '_flag' for s in features_name]
    data_use_final = imputation_median_indicators(data_use_final, features_name, features_name_flag, data_name, data_save_path)

    # [3] get BMI, shock index, bun_creatinine, egfr, GNRI, nlr info
    data_use_final['bmi'] = 10000*data_use_final['weight']/(data_use_final['height']**2)
    data_use_final['bmi'] = data_use_final['bmi'].round(2)

    data_use_final['shock_index'] = (data_use_final['heart_rate_mean']/data_use_final['sbp_mean']).round(2)
    data_use_final['bun_creatinine'] = (data_use_final['bun_max'] / data_use_final['creatinine_max']).round(2)

    # egfr: gender, creatinine_max, age, ethnicity
    egfr = pd.DataFrame()
    egfr = data_use_final[['id', 'gender', 'age', 'ethnicity', 'creatinine_max']]
    egfr['egfr'] = 186*(egfr['creatinine_max'].pow(-1.154))*(egfr['age'].pow(-0.203))
    egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] == 1), 'egfr'] = 0.742*1.210*egfr['egfr']
    egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] != 1), 'egfr'] = 0.742*egfr['egfr']
    egfr['egfr'] = egfr['egfr'].round(2)

    # ideal weight = height (cm) - 100 - ([height(cm) - 150]/4) for men
    # ideal weight = height (cm) - 100 - ([height(cm) - 150]/2.5) for women
    # GNRI = [14.89*albumin(g/dL)] + [41.7*(weight/ideal weight)]
    gnri = pd.DataFrame()
    gnri = data_use_final[['id', 'gender', 'albumin_min', 'weight', 'height']]
    gnri['ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/4)
    gnri.loc[gnri['gender'] == 0, 'ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/2.5)
    gnri['gnri'] = (14.89*gnri['albumin_min']) + (41.7*(gnri['weight']/gnri['ideal_weight']))
    gnri['gnri'] = gnri['gnri'].round(2)

    nlr = pd.DataFrame() # nlr: Neutrophil-to-Lymphocyte Ratio
    nlr = data_use_final[['id', 'neutrophils_min', 'lymphocytes_min', 'lymphocytes_min_flag']]
    nlr['nlr_flag'] = nlr['lymphocytes_min_flag']
    nlr['nlr'] = 0
    nlr.loc[nlr['lymphocytes_min_flag'] > 0, 'nlr'] = nlr['neutrophils_min']/nlr['lymphocytes_min']
    nlr['nlr'] = nlr['nlr'].round(2)

    # [4] merge: main part + gnri + nlr
    data_use_final.drop(['ethnicity', 'electivesurgery', 'height', 'weight'], axis=1, inplace=True, errors='ignore') # not be considered in the study
    data_use_final = pd.merge(data_use_final, gnri[['id', 'gnri']], on='id')
    data_use_final = pd.merge(data_use_final, egfr[['id', 'egfr']], on='id')
    data_use_final = pd.merge(data_use_final, nlr[['id', 'nlr', 'nlr_flag']], on='id')
    # data_use_final = pd.merge(data_use_final, data[['id', 'sofa', 'oasis', 'saps', 'apsiii', 'oasis_prob', 'saps_prob', 'sofa_prob', 'apsiii_prob']], on='id')

    # [5] special considering parts of features with high missing ratio
    #         impute empty with 0
    if data_name == 'ams_use':
        data_use_final.fillna(0, inplace=True)


    # [6] category features reset dtype to int
    columns_int_names = ['activity_bed', 'activity_sit', 'activity_stand', 'admission_type', \
                        'code_status', 'death_hosp', 'dobutamine', 'dopamine', 'epinephrine', \
                        'gender', 'norepinephrine', 'vent']
    columns_int_names = columns_int_names + [s for s in data_use_final.columns.values.tolist() if '_flag' in s]
    data_use_final[columns_int_names] = data_use_final[columns_int_names].astype(int)

    # [7] no need study columns - drop again
    data_use_final.drop(['albumin_min', 'albumin_min_flag'], axis=1, inplace=True)

    # save
    data_use_final.to_csv(data_save_path + data_name + '.csv', index=False)


def generate_data_imputation_miceforest(data_all, data_save_path):
    """
    :param data_all: all datasets' dictionary {'MIMIC_eICU_use', 'eICU_extra', 'ams_use', 'MIMIC_temporal'}
    :param data_save_path: imputed data save path
    :return data_use_final: directly save the imputed data
    """ 
    # [1] split to impute and non-impute, save
    data_use = {}
    for i in ['MIMIC_eICU_use', 'eICU_extra', 'ams_use', 'MIMIC_temporal']:
        data_impute, data_non_impute = pd.DataFrame(), pd.DataFrame()

        list_nonimpute_name = [
        'code_status', 'code_status_eva_flag', 'death_hosp', 'activity_eva_flag',
        'activity_bed', 'activity_sit', 'activity_stand', 'admission_type', 
        'dobutamine', 'dopamine', 'electivesurgery', 'epinephrine', 
        'ethnicity', 'gender', 'norepinephrine', 'vent', 'id']
        data_non_impute = data_all[i][list_nonimpute_name]

        list_impute_name = list(set(data_all[i].columns.to_list()) - set(list_nonimpute_name))
        data_impute = data_all[i][list_impute_name + ['id']]

        if i == 'ams_use':
            data_impute['cci_score'] = np.nan
            data_impute['pre_icu_los_day'] = np.nan
        data_impute[list_impute_name] = data_impute[list_impute_name].astype(float)
        data_impute = data_impute.reindex(sorted(data_impute.columns), axis=1)

        if i == 'MIMIC_eICU_use':
            data_impute_dev, data_impute_cal_val = pd.DataFrame(), pd.DataFrame()
            data_impute_dev, data_impute_cal_val = train_test_split(data_impute, test_size=0.2, random_state=0, shuffle=True)
            data_impute = {'dev':data_impute_dev, 'cal_val':data_impute_cal_val}
            del data_impute_dev, data_impute_cal_val
        
        data_use[i] = {'impute_before':data_impute, 'nonimpute':data_non_impute}

    # [2] imputation using miceforest
    # Create kernel and imputation.
    kernel = mf.ImputationKernel(data_use['MIMIC_eICU_use']['impute_before']['dev'], datasets=4, save_all_iterations=True, save_models=1, random_state=2)
    kernel.mice(iterations=3, n_jobs=-1)
    data_use['MIMIC_eICU_use']['imputed'] = {}
    data_use['MIMIC_eICU_use']['imputed']['dev'] = {}
    for m in range(4):
        data_use['MIMIC_eICU_use']['imputed']['dev'][m] = kernel.complete_data(m)
    for j in ['MIMIC_eICU_use', 'eICU_extra', 'ams_use', 'MIMIC_temporal']:
        if j == 'MIMIC_eICU_use':
            kernel_new = kernel.impute_new_data(data_use[j]['impute_before']['cal_val'])
            data_use[j]['imputed']['cal_val'] = {}
            # merge mimic_eicu (dev & cal_val) and rank by id
            for m in range(4):
                data_use[j]['imputed']['cal_val'][m] = kernel_new.complete_data(m)
                data_use[j]['imputed'][m] = data_use[j]['imputed']['dev'][m].append(data_use[j]['imputed']['cal_val'][m], ignore_index=True)
                data_use[j]['imputed'][m].sort_values('id', inplace = True)
            # drop mimic_eicu no need dict (dev & cal_val)
            for key in ['dev', 'cal_val']:
                del data_use[j]['imputed'][key]
        else:
            kernel_new = kernel.impute_new_data(data_use[j]['impute_before'])
            data_use[j]['imputed'] = {}
            for m in range(4):
                data_use[j]['imputed'][m] = kernel_new.complete_data(m)

    # [3] change type and merge label
    for i in ['MIMIC_eICU_use', 'eICU_extra', 'ams_use', 'MIMIC_temporal']:
        data_average = pd.DataFrame()
        for m in range(4):
            data_each = pd.DataFrame()
            data_each = data_use[i]['imputed'][m]
            data_each[['cci_score', 'fio2_max', 'gcs_min']] = data_each[['cci_score', 'fio2_max', 'gcs_min']].round(0).astype(int)
            data_average = data_average.append(data_each)
        data_average = data_average.groupby(['id']).mean().reset_index()
        data_average.loc[data_average['fio2_max'] > 100, 'fio2_max'] = 100
        data_average.loc[data_average['fio2_max'] < 21, 'fio2_max'] = 21
        data_average.loc[data_average['gcs_min'] > 15, 'gcs_min'] = 15
        data_average.loc[data_average['gcs_min'] < 3, 'gcs_min'] = 3
        data_average = pd.merge(data_average, data_use[i]['nonimpute'].astype('Int64'), on='id')

        if i == 'ams_use':
            data_average.fillna(0, inplace=True)

        # [4] get BMI, shock index, bun_creatinine, egfr, GNRI, nlr info
        data_average['bmi'] = 10000*data_average['weight']/(data_average['height']**2)
        data_average['bmi'] = data_average['bmi'].round(2)

        data_average['shock_index'] = (data_average['heart_rate_mean']/data_average['sbp_mean']).round(2)
        data_average['bun_creatinine'] = (data_average['bun_max'] / data_average['creatinine_max']).round(2)

        # egfr: gender, creatinine_max, age, ethnicity
        egfr = pd.DataFrame()
        egfr = data_average[['id', 'gender', 'age', 'ethnicity', 'creatinine_max']]
        egfr['egfr'] = 186*(egfr['creatinine_max'].pow(-1.154))*(egfr['age'].pow(-0.203))
        egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] == 1), 'egfr'] = 0.742*1.210*egfr['egfr']
        egfr.loc[(egfr['gender'] == 0) & (egfr['ethnicity'] != 1), 'egfr'] = 0.742*egfr['egfr']
        egfr['egfr'] = egfr['egfr'].round(2)

        # ideal weight = height (cm) - 100 - ([height(cm) - 150]/4) for men
        # ideal weight = height (cm) - 100 - ([height(cm) - 150]/2.5) for women
        # GNRI = [14.89*albumin(g/dL)] + [41.7*(weight/ideal weight)]
        gnri = pd.DataFrame()
        gnri = data_average[['id', 'gender', 'albumin_min', 'weight', 'height']]
        gnri['ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/4)
        gnri.loc[gnri['gender'] == 0, 'ideal_weight'] = gnri['height'] - 100 - ((gnri['height'] - 150)/2.5)
        gnri['gnri'] = (14.89*gnri['albumin_min']) + (41.7*(gnri['weight']/gnri['ideal_weight']))
        gnri['gnri'] = gnri['gnri'].round(2)

        nlr = pd.DataFrame() # nlr: Neutrophil-to-Lymphocyte Ratio
        nlr = data_average[['id', 'neutrophils_min', 'lymphocytes_min']]
        nlr['nlr'] = 0
        nlr.loc[nlr['lymphocytes_min'] > 0, 'nlr'] = nlr['neutrophils_min']/nlr['lymphocytes_min']
        nlr['nlr'] = nlr['nlr'].round(2)

        data_average.drop(['ethnicity', 'electivesurgery', 'height', 'weight'], axis=1, inplace=True, errors='ignore') # not be considered in the study
        data_average = pd.merge(data_average, gnri[['id', 'gnri']], on='id')
        data_average = pd.merge(data_average, egfr[['id', 'egfr']], on='id')
        data_average = pd.merge(data_average, nlr[['id', 'nlr']], on='id')

        # [5] category features reset dtype to int
        columns_int_names = ['activity_bed', 'activity_sit', 'activity_stand', 'admission_type', \
                            'code_status', 'death_hosp', 'dobutamine', 'dopamine', 'epinephrine', \
                            'gender', 'norepinephrine', 'vent', 'activity_eva_flag', 'code_status_eva_flag',\
                            'cci_score', 'fio2_max', 'gcs_min']
        data_average[columns_int_names] = data_average[columns_int_names].astype(int)

        # [6] no need study columns - drop again
        data_average.drop(['albumin_min'], axis=1, inplace=True)

        # [7] save the imputation data
        data_average.to_csv(data_save_path + i + '.csv', index=False)


def diff_data_type(data_initial, needing_type, outcome_name, imbalance_med):
    """
    :param data_initial: initial data without processing containing id info
    :para needing_type: 1 (lr, svm, nn type) or 2 (rf, dt, xgb)
    :param outcome_name: like 'death_hosp'
    :param imbalance_med: 'nonprocess', 'upsampling', 'downsampling'
    :return data: without considering the group label
    """    
    # get different needing training and test data
    #     1.  type1 -- fit to regression model : LR, SVM      #
    if needing_type == 1:
        data_type1 = pd.DataFrame()
        X_type1, y_type1 = pd.DataFrame(), pd.DataFrame()

        data_type1 = data_initial.copy()
        data_type1 = data_initial.drop(['id'], axis = 1)
        # one-hot processing and normalization
        names_dummies_all = ['Code status', 'Code status (eva)', 'Activity (eva)', 'Activity (bed)', 
        'Activity (sit)', 'Activity (stand)', 'Admission type', 
        'Dobutamine', 'Dopamine', 'Epinephrine', 'Gender', 'Norepinephrine', 'Ventilation'] 
        names_dummies = list(set(names_dummies_all) & set(data_type1.columns.to_list())) # acquire the existing columns

        for i in range(len(names_dummies)):
            vars()['dummies_' + names_dummies[i]] = pd.get_dummies(data_type1[names_dummies[i]],
                                                                   prefix=names_dummies[i])
            # https://www.daniweb.com/programming/software-development/threads/111526/setting-a-string-as-a-variable-name
            data_type1 = pd.concat([data_type1, vars()['dummies_' + names_dummies[i]]], axis=1)
            del vars()['dummies_' + names_dummies[i]]
        # data_type_new.drop(names_dummies, axis=1, inplace=True)
        data_type1.drop(names_dummies, axis=1, inplace=True)
        X_type1 = data_type1.drop(outcome_name, axis=1)  # type1 features
        X_type1 = X_type1.reindex(sorted(X_type1.columns), axis=1)  # sort features name
        y_type1 = data_type1[outcome_name]  # type1 target
        X_train, X_test_all, y_train, y_test_all = train_test_split(X_type1, y_type1, test_size=0.2, random_state=0, shuffle=True)
        X_cal, X_test, y_cal, y_test = train_test_split(X_test_all, y_test_all, test_size=0.5, random_state=0, shuffle=True)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
        X_cal = pd.DataFrame(min_max_scaler.fit_transform(X_cal))
        X_test = pd.DataFrame(min_max_scaler.fit_transform(X_test))
        X_train.columns = X_type1.columns
        X_cal.columns = X_type1.columns
        X_test.columns = X_type1.columns

    else:
        #      2.  type2 -- fit to tree model: Xgboost, RF, DF      #
        data_type2 = pd.DataFrame().copy()
        X_type2, y_type2 = pd.DataFrame(), pd.DataFrame()
        data_type2 = data_initial
        data_type2 = data_initial.drop(['id'], axis = 1)
        X_type2 = data_type2.drop(outcome_name, axis=1)  # type1 features
        X_type2 = X_type2.reindex(sorted(X_type2.columns), axis=1)  # sort features name
        y_type2 = data_type2[outcome_name]
        X_train, X_test_all, y_train, y_test_all = train_test_split(X_type2, y_type2, test_size=0.2, random_state=0, shuffle=True)
        X_cal, X_test, y_cal, y_test = train_test_split(X_test_all, y_test_all, test_size=0.5, random_state=0, shuffle=True)

    id_train, id_test_all = train_test_split(data_initial['id'], test_size=0.2, random_state=0, shuffle=True)
    id_cal, id_test = train_test_split(id_test_all, test_size=0.5, random_state=0, shuffle=True)

    # imbalance data solving methods
    if imbalance_med == 'upsampling':
        ROS = RandomOverSampler(random_state=0)
        X_train['id'] = id_train
        X_train, y_train = ROS.fit_resample(X_train, y_train)
        id_train = X_train['id']
        X_train.drop(['id'], inplace=True, axis=1)
    elif imbalance_med == 'downsampling':
        RUS = RandomUnderSampler(random_state=0)
        X_train['id'] = id_train
        X_train, y_train = RUS.fit_resample(X_train, y_train)
        id_train = X_train['id']
        X_train.drop(['id'], inplace=True, axis=1)
    else:
        X_train, y_train, id_train = X_train, y_train, id_train


    X_train.reset_index(drop=True, inplace=True)
    X_cal.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_cal.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    id_train.reset_index(drop=True, inplace=True)
    id_cal.reset_index(drop=True, inplace=True)
    id_test.reset_index(drop=True, inplace=True)    

    data = {}
    data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
        'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
        'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}    

    return data


def get_train_cal_test_data(MIMIC_eICU_use, eICU_extra, ams_use, MIMIC_temporal, train_test_name, imbalance_med):
    data = {} # output
    MIMIC_eICU_use = MIMIC_eICU_use.reindex(sorted(MIMIC_eICU_use.columns), axis=1)
    eICU_extra = eICU_extra.reindex(sorted(eICU_extra.columns), axis=1)
    ams_use = ams_use.reindex(sorted(ams_use.columns), axis=1)
    MIMIC_temporal = MIMIC_temporal.reindex(sorted(MIMIC_temporal.columns), axis=1)

    # 1. get development set
    development_set = pd.DataFrame()
    development_set = MIMIC_eICU_use
    data_type1_dev, data_type2_dev = {}, {}
    data_type1_dev = diff_data_type(development_set, 1, 'death_hosp', imbalance_med)
    data_type2_dev = diff_data_type(development_set, 2, 'death_hosp', imbalance_med)

    # 2. get internal or temporal or external set
    evaluation_set = pd.DataFrame()
    if train_test_name == 'MIMIC_eICU-Ams': # internal validation
        evaluation_set = ams_use
    elif train_test_name == 'MIMIC_eICU-MIMIC': # temporal validation
        evaluation_set = MIMIC_temporal
    elif train_test_name == 'MIMIC_eICU-eICU': # external validation
        evaluation_set = eICU_extra
    else: # external validation
        evaluation_set = pd.DataFrame()
    data_type1_eva, data_type2_eva = {}, {}
    if train_test_name == 'MIMIC_eICU-MIMIC_eICU':
        data_type1_eva = {}
        data_type2_eva = {}
    else:
        data_type1_eva = diff_data_type(evaluation_set, 1, 'death_hosp', 'nonprocess')
        data_type2_eva = diff_data_type(evaluation_set, 2, 'death_hosp', 'nonprocess')

    # 3. get the output data
    X_train_type1, X_cal_type1, X_test_type1 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y_train_type1, y_cal_type1, y_test_type1 = pd.Series([]), pd.Series([]), pd.Series([])
    X_train_type2, X_cal_type2, X_test_type2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y_train_type2, y_cal_type2, y_test_type2 = pd.Series([]), pd.Series([]), pd.Series([])    
    id_train, id_cal, id_test = pd.Series([]), pd.Series([]), pd.Series([])
    if train_test_name == 'MIMIC_eICU-MIMIC_eICU':
        X_train_type1, X_cal_type1, X_test_type1 = data_type1_dev['X_train'], data_type1_dev['X_cal'], data_type1_dev['X_test']
        y_train_type1, y_cal_type1, y_test_type1 = data_type1_dev['y_train'], data_type1_dev['y_cal'], data_type1_dev['y_test']
        X_train_type2, X_cal_type2, X_test_type2 = data_type2_dev['X_train'], data_type2_dev['X_cal'], data_type2_dev['X_test']
        y_train_type2, y_cal_type2, y_test_type2 = data_type2_dev['y_train'], data_type2_dev['y_cal'], data_type2_dev['y_test']
        id_train, id_cal, id_test = data_type2_dev['id_train'], data_type2_dev['id_cal'], data_type2_dev['id_test']
        # check the missing one-hot labeled columns and add them
        if len(list(set(X_train_type1.columns.values.tolist()) - set(X_test_type1.columns.values.tolist()))) > 0:
            for col in list(set(X_train_type1.columns.values.tolist()) - set(X_test_type1.columns.values.tolist())):
                # missing activity info (will drop when correct those columns)
                X_test_type1[col] = 0
    elif train_test_name == 'MIMIC_eICU-eICU':
        X_train_type1, X_cal_type1 = data_type1_dev['X_train'], data_type1_eva['X_cal']
        X_test_type1 = pd.concat([data_type1_eva['X_train'], data_type1_eva['X_test']], axis=0, ignore_index=True)
        if len(list(set(X_train_type1.columns.values.tolist()) - set(X_test_type1.columns.values.tolist()))) > 0:
            for col in list(set(X_train_type1.columns.values.tolist()) - set(X_test_type1.columns.values.tolist())):
                # missing activity info (will drop when correct those columns)
                X_test_type1[col] = 0
        if len(list(set(X_train_type1.columns.values.tolist()) - set(X_cal_type1.columns.values.tolist()))) > 0:
            for col in list(set(X_train_type1.columns.values.tolist()) - set(X_cal_type1.columns.values.tolist())):
                # missing activity info (will drop when correct those columns)
                X_cal_type1[col] = 0        
        X_cal_type1 = X_cal_type1[X_train_type1.columns.values.tolist()]
        X_test_type1 = X_test_type1[X_train_type1.columns.values.tolist()]
        y_train_type1, y_cal_type1 = data_type1_dev['y_train'], data_type1_eva['y_cal']
        y_test_type1 = pd.concat([data_type1_eva['y_train'], data_type1_eva['y_test']], axis=0, ignore_index=True)
        X_train_type2, X_cal_type2 = data_type2_dev['X_train'], data_type2_eva['X_cal']
        X_test_type2 = pd.concat([data_type2_eva['X_train'], data_type2_eva['X_test']], axis=0, ignore_index=True)
        X_cal_type2 = X_cal_type2[X_train_type2.columns.values.tolist()]
        X_test_type2 = X_test_type2[X_train_type2.columns.values.tolist()]
        y_train_type2, y_cal_type2 = data_type2_dev['y_train'], data_type2_eva['y_cal']
        y_test_type2 = pd.concat([data_type2_eva['y_train'], data_type2_eva['y_test']], axis=0, ignore_index=True)
        id_train, id_cal = data_type2_dev['id_train'], data_type2_eva['id_cal']
        id_test = pd.concat([data_type2_eva['id_train'], data_type2_eva['id_test']], axis=0, ignore_index=True)
    else:
        X_train_type1, X_test_type1 = data_type1_dev['X_train'], data_type1_eva['X_train']
        X_cal_type1 = pd.concat([data_type1_eva['X_cal'], data_type1_eva['X_test']], axis=0, ignore_index=True)
        if len(list(set(X_train_type1.columns.values.tolist()) - set(X_test_type1.columns.values.tolist()))) > 0:
            for col in list(set(X_train_type1.columns.values.tolist()) - set(X_test_type1.columns.values.tolist())):
                # missing activity info (will drop when correct those columns)
                X_test_type1[col] = 0
        if len(list(set(X_train_type1.columns.values.tolist()) - set(X_cal_type1.columns.values.tolist()))) > 0:
            for col in list(set(X_train_type1.columns.values.tolist()) - set(X_cal_type1.columns.values.tolist())):
                # missing activity info (will drop when correct those columns)
                X_cal_type1[col] = 0
        X_cal_type1 = X_cal_type1[X_train_type1.columns.values.tolist()]
        X_test_type1 = X_test_type1[X_train_type1.columns.values.tolist()]
        y_train_type1, y_test_type1 = data_type1_dev['y_train'], data_type1_eva['y_train']
        y_cal_type1 = pd.concat([data_type1_eva['y_cal'], data_type1_eva['y_test']], axis=0, ignore_index=True)                

        X_train_type2, X_test_type2 = data_type2_dev['X_train'], data_type2_eva['X_train']
        X_cal_type2 = pd.concat([data_type2_eva['X_cal'], data_type2_eva['X_test']], axis=0, ignore_index=True)
        X_cal_type2 = X_cal_type2[X_train_type2.columns.values.tolist()]
        X_test_type2 = X_test_type2[X_train_type2.columns.values.tolist()]
        y_train_type2, y_test_type2 = data_type2_dev['y_train'], data_type2_eva['y_train']
        y_cal_type2 = pd.concat([data_type2_eva['y_cal'], data_type2_eva['y_test']], axis=0, ignore_index=True)

        id_train, id_test = data_type2_dev['id_train'], data_type2_eva['id_train']
        id_cal = pd.concat([data_type2_eva['id_cal'], data_type2_eva['id_test']], axis=0, ignore_index=True)
            
    del data_type1_dev, data_type2_dev, data_type1_eva, data_type2_eva
    

    data = {
        'data_need_type1': {'X_train': X_train_type1, 'X_cal': X_cal_type1, 'X_test': X_test_type1, \
                            'y_train': y_train_type1, 'y_cal': y_cal_type1, 'y_test': y_test_type1, \
                            'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test},
        'data_need_type2': {'X_train': X_train_type2, 'X_cal': X_cal_type2, 'X_test': X_test_type2, \
                            'y_train': y_train_type2, 'y_cal': y_cal_type2, 'y_test': y_test_type2, \
                            'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test},
        }

    return data


def generate_xgb_gossis_sms(xgb_path, gossis_path, sms_path, path_save):

    # acquire the basic information
    data_basic_path = ''
    data_basic_path = 'D:/project/older_score/data/model_use/miceforest_lgb/'
    MIMIC_eICU_use = pd.read_csv(data_basic_path + 'MIMIC_eICU_use.csv')[['id', 'age', 'gender']]
    eICU_extra = pd.read_csv(data_basic_path + 'eICU_extra.csv')[['id', 'age', 'gender']]
    ams_use = pd.read_csv(data_basic_path + 'ams_use.csv')[['id', 'age', 'gender']]
    MIMIC_temporal = pd.read_csv(data_basic_path + 'MIMIC_temporal.csv')[['id', 'age', 'gender']]
    data_basic_path = ''
    data_basic_path = 'D:/project/older_score/data/'
    mimic = pd.read_csv(data_basic_path + 'mimic.csv')[['id', 'ethnicity']]
    eicu = pd.read_csv(data_basic_path + 'eicu.csv')[['id', 'ethnicity']]
    mimic_eicu = pd.concat([mimic, eicu], axis=0)
    mimic_eicu.reset_index(drop=True, inplace=True)
    MIMIC_eICU_use = pd.merge(MIMIC_eICU_use, mimic_eicu, on='id')
    eICU_extra = pd.merge(eICU_extra, mimic_eicu, on='id')
    ams_use['ethnicity'] = 'unknown'
    MIMIC_temporal = pd.merge(MIMIC_temporal, mimic, on='id')
    del mimic, eicu, mimic_eicu

    data_basic_path = ''
    data_basic_path = sms_path
    mimic_sms, eicu_sms, ams_sms, mimic_eicu_sms = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mimic_sms = pd.read_csv(data_basic_path + 'mimic_sms.csv')[['id', 'sms_prob']]
    eicu_sms = pd.read_csv(data_basic_path + 'eicu_sms.csv')[['id', 'sms_prob']]
    ams_sms = pd.read_csv(data_basic_path + 'ams_sms.csv')[['id', 'sms_prob']]
    mimic_eicu_sms = pd.concat([mimic_sms, eicu_sms], axis=0)
    mimic_eicu_sms.reset_index(drop=True, inplace=True)
    del mimic_sms, eicu_sms

    # merge the xgb, gossis, sms
    MIMIC_eICU_need, eICU_need, Ams_need, MIMIC_need = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    MIMIC_eICU_need = pd.read_csv(xgb_path + 'MIMIC_eICU-MIMIC_eICU/' + 'models_pred_true.csv')[['id', 'true_label', 'xgb']]
    MIMIC_eICU_need = pd.merge(MIMIC_eICU_need, mimic_eicu_sms, on='id')
    MIMIC_eICU_need = pd.merge(MIMIC_eICU_need, MIMIC_eICU_use, on='id')
    MIMIC_eICU_need.loc[MIMIC_eICU_need['age'] >= 80, 'age_group'] = 'old-old' 
    MIMIC_eICU_need.loc[MIMIC_eICU_need['age'] < 80, 'age_group'] = 'young-old'
    MIMIC_eICU_need.loc[MIMIC_eICU_need['gender'] == 0, 'gender_group'] = 'female'
    MIMIC_eICU_need.loc[MIMIC_eICU_need['gender'] == 1, 'gender_group'] = 'male'    

    eICU_need = pd.read_csv(xgb_path + 'MIMIC_eICU-eICU/' + 'models_pred_true.csv')[['id', 'true_label', 'xgb']]
    eicu_gossis = pd.read_csv(gossis_path + 'eicu_gossis.csv')[['id', 'gossis']]
    eICU_need = pd.merge(eICU_need, mimic_eicu_sms, on='id')
    eICU_need = eICU_need.merge(eicu_gossis, on='id', how='left')
    eICU_need['gossis'] = eICU_need['gossis'].fillna(eICU_need['gossis'].median())
    eICU_need = pd.merge(eICU_need, eICU_extra, on='id')
    eICU_need.loc[eICU_need['age'] >= 80, 'age_group'] = 'old-old' 
    eICU_need.loc[eICU_need['age'] < 80, 'age_group'] = 'young-old'
    eICU_need.loc[eICU_need['gender'] == 0, 'gender_group'] = 'female'
    eICU_need.loc[eICU_need['gender'] == 1, 'gender_group'] = 'male'
    del eicu_gossis

    Ams_need = pd.read_csv(xgb_path + 'MIMIC_eICU-Ams/' + 'models_pred_true.csv')[['id', 'true_label', 'xgb']]
    Ams_need = pd.merge(Ams_need, ams_sms, on='id')
    Ams_need = pd.merge(Ams_need, ams_use, on='id')
    Ams_need.loc[Ams_need['age'] >= 80, 'age_group'] = 'old-old' 
    Ams_need.loc[Ams_need['age'] < 80, 'age_group'] = 'young-old'
    Ams_need.loc[Ams_need['gender'] == 0, 'gender_group'] = 'female'
    Ams_need.loc[Ams_need['gender'] == 1, 'gender_group'] = 'male'

    MIMIC_need = pd.read_csv(xgb_path + 'MIMIC_eICU-MIMIC/' + 'models_pred_true.csv')[['id', 'true_label', 'xgb']]
    MIMIC_need = pd.merge(MIMIC_need, mimic_eicu_sms, on='id')
    MIMIC_need = pd.merge(MIMIC_need, MIMIC_temporal, on='id')
    MIMIC_need.loc[MIMIC_need['age'] >= 80, 'age_group'] = 'old-old' 
    MIMIC_need.loc[MIMIC_need['age'] < 80, 'age_group'] = 'young-old'
    MIMIC_need.loc[MIMIC_need['gender'] == 0, 'gender_group'] = 'female'
    MIMIC_need.loc[MIMIC_need['gender'] == 1, 'gender_group'] = 'male'    

    MIMIC_eICU_need.to_csv(path_save + 'MIMIC_eICU_need.csv', index=False)
    eICU_need.to_csv(path_save + 'eICU_need.csv', index=False)
    Ams_need.to_csv(path_save + 'Ams_need.csv', index=False)
    MIMIC_need.to_csv(path_save + 'MIMIC_need.csv', index=False)