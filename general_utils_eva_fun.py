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
# import train_models as tmm
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


# get all needs metrics of performance
def model_performance_params(data, data_pred_proba, ts_use, ts_value):
    """
    data: the truth label of target [array]
    data_pred_proba: predict probability of target with one columns [array]
    ts_use: 'True' or 'False' (if true, will use ts_value, else will not use ts_value) [Bool]
    ts_value: float value (if ts_use = 'True', will use it - input the value needed, or not use it)

    """
    fpr, tpr, thresholds_ROC = roc_curve(data, data_pred_proba)
    precision, recall, thresholds = precision_recall_curve(data, data_pred_proba)
    average_precision = average_precision_score(data, data_pred_proba)
    brier_score = brier_score_loss(data, data_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)

    threshold_final = []
    if ts_use == 'False':
        optimal_idx = []
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_ROC[optimal_idx]
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_pred_proba))
        data_pred[data_pred_proba >= optimal_threshold] = 1
        threshold_final = optimal_threshold
    else:
        optimal_idx = []
        optimal_idx = np.max(np.where(thresholds_ROC >= ts_value))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_pred_proba))
        data_pred[data_pred_proba >= ts_value] = 1
        threshold_final = ts_value

    tn, fp, fn, tp = confusion_matrix(data, data_pred).ravel()
    accuracy = accuracy_score(data, data_pred)
    F1 = f1_score(data, data_pred)  # not consider the imbalance, using 'binary' 2tp/(2tp+fp+fn)
    precision_c = tp/(tp+fp)

    parameters = {'auc': roc_auc, 'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy,
                  'F1': F1, 'precision': precision_c, 'ap':average_precision, 'brier_score': brier_score, 'threshold': threshold_final}
    roc_plot_data = {'fpr_data': fpr, 'tpr_data': tpr}
    return parameters, roc_plot_data


# get the index of all values using booststrap
# 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
def model_performance_params_bootstrap(data, data_ths, num_iterations):
    """
    input
    data: dataframe ('true_label', 'probability')
    data_ths: float -- threshold of using
    num_iterations: int -- the iteration time
    output
    stats: dataframe: 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
    """
    n_iterations = num_iterations
    n_size = int(data.shape[0] * 0.80)
    stats = list()
    for i in range(n_iterations):
        data_use = resample(data.values, n_samples=n_size)
        fpr, tpr, thresholds_ROC = roc_curve(data_use['true_label'], data_use['probability'])
        precision, recall, thresholds = precision_recall_curve(data_use['true_label'], data_use['probability'])
        average_precision = average_precision_score(data_use['true_label'], data_use['probability'])
        brier_score = brier_score_loss(data_use['true_label'], data_use['probability'], pos_label=1)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.max(np.where(thresholds_ROC >= data_ths))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_use))
        data_pred[data_use['probability'] >= data_ths] = 1
        tn, fp, fn, tp = confusion_matrix(data_use['true_label'], data_pred).ravel()
        accuracy = accuracy_score(data_use['true_label'], data_pred)
        F1 = f1_score(data_use['true_label'], data_pred)
        precision_c = tp / (tp + fp)
        npv = (tn)/(tn+fn)
        score = []
        score = [roc_auc, sensitivity, specificity, accuracy, F1, precision_c, npv, average_precision]
        stats.append(score)
    stats = pd.DataFrame.from_records(stats)
    stats.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    
    return stats


# get the index with 95% CI
# 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier'
def model_performance_params_bootstrap_95CI(data, data_ths, num_iterations):
    """
    input
    data: dataframe ('true_label', 'probability')
    data_ths: float -- threshold of using
    num_iterations: int -- the iteration time
    output
    stats_new: dataframe -- 95% CI: 'roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score'
    """
    n_iterations = num_iterations
    n_size = int(data.shape[0] * 0.90)
    stats = list()
    for i in range(n_iterations):
        data_use = resample(data.values, n_samples=n_size)
        data_use = pd.DataFrame(data_use, columns=['true_label', 'probability'])
        data_use['true_label'] = data_use['true_label'].astype(int)
        fpr, tpr, thresholds_ROC = roc_curve(data_use['true_label'], data_use['probability'])
        precision, recall, thresholds = precision_recall_curve(data_use['true_label'], data_use['probability'])
        average_precision = average_precision_score(data_use['true_label'], data_use['probability'])
        brier_score = brier_score_loss(data_use['true_label'], data_use['probability'], pos_label=1)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.max(np.where(thresholds_ROC >= data_ths))
        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]
        data_pred = np.zeros(len(data_use))
        data_pred[data_use['probability'] >= data_ths] = 1
        if sum(data_use['true_label']) == 0:
            precision_c, npv = np.nan, np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(data_use['true_label'], data_pred).ravel()
            precision_c = tp / (tp + fp)
            npv = (tn) / (tn + fn)
        accuracy = accuracy_score(data_use['true_label'], data_pred)
        F1 = f1_score(data_use['true_label'], data_pred)
        score = []
        score = [roc_auc, sensitivity, specificity, accuracy, F1, precision_c, npv, average_precision, brier_score]
        stats.append(score)
    stats = pd.DataFrame.from_records(stats)
    stats.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    stats = stats.dropna()
    # calculate 95% CI of each model
    alpha = 0.95
    p_l = ((1.0-alpha)/2.0) * 100
    p_u = (alpha+((1.0-alpha)/2.0)) * 100

    stats_new = list()
    for j in range(9): # auc, sen, spe, acc, f1, pre, npv, ap, brier_score
        lower = max(0.0, round(np.percentile(list(stats.iloc[:,j]), p_l),3))
        upper = min(1.0, round(np.percentile(list(stats.iloc[:,j]), p_u),3))
        val = round(np.percentile(list(stats.iloc[:,j]), 50),3)
        val = str(val) + ' (' + str(lower) + '-' + str(upper) + ')'
        stats_new.append(val)
    stats_new = pd.DataFrame.from_records([stats_new])
    stats_new.columns = ['roc_auc', 'sensitivity', 'specificity', 'accuracy', 'F1', 'precision', 'npv', 'ap', 'brier_score']
    
    return stats, stats_new


def score_performance_cal(X_test_score, y_test_score, score_name, ts_use, ts_value):
    para_score, roc_plot_score = model_performance_params(y_test_score, X_test_score[score_name], ts_use, ts_value)
    parameters = [score_name, round(para_score['auc'],3),
                   round(para_score['sensitivity'],3), round(para_score['specificity'],3), round(para_score['accuracy'],3),
                   round(para_score['F1'],3), round(para_score['precision'],3), round(para_score['ap'],3), 
                   round(para_score['brier_score'],3), para_score['threshold']]

    return parameters