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
import general_utils_eva_fun as ef 


def older_lr_model(data, drop_cols_name, model_info):
    # Logistic regression
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 0. define the variables
    predicted_LR, probas_LR, para_LR_bs, roc_plot_LR_bs, parameters = [], [], {}, {}, {} 
    fpr_LR, tpr_LR, threshold_LR = [], [], []
    X_train_type1, X_cal_type1, X_test_type1, y_train_type1, y_cal_type1, y_test_type1 = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([]), pd.Series([])
    # 1. load the data for usage and drop no needs columns
    X_train_type1, X_cal_type1, X_test_type1, y_train_type1, y_cal_type1, y_test_type1 = \
    data['X_train'].drop(drop_cols_name, axis=1), data['X_cal'].drop(drop_cols_name, axis=1), \
    data['X_test'].drop(drop_cols_name, axis=1), data['y_train'], data['y_cal'], data['y_test']
    
    # 2. train model
    if model_info['model_use'] == 'False':
        clf_LR_bs = LogisticRegression(multi_class="ovr", penalty="l1", class_weight="balanced", solver="liblinear")
        re_LR_bs = clf_LR_bs.fit(X_train_type1, y_train_type1)
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = re_LR_bs 
            re_LR_bs = CalibratedClassifierCV(re_LR_bs, method='isotonic', cv='prefit')
            re_LR_bs.fit(X_cal_type1, y_cal_type1)
            model_use = re_LR_bs
        else:
            model_middle = re_LR_bs
            model_use = re_LR_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # '.../no_cal.dat', '.../no_cal.dat'
    else:
        re_LR_bs = pickle.load(open(model_info['model_path_full_info'], "rb"))
        if model_info['cal_yn'] == 'True':
            model_middle = re_LR_bs 
            re_LR_bs = CalibratedClassifierCV(re_LR_bs, method='isotonic', cv='prefit')
            re_LR_bs.fit(X_cal_type1, y_cal_type1)
            model_use = re_LR_bs
        else:
            model_middle = re_LR_bs
            model_use = re_LR_bs        
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))

    # 4. get performance results    
    predicted_LR = model_use.predict(X_test_type1)
    probas_LR = model_use.predict_proba(X_test_type1)
    para_LR_bs, roc_plot_LR_bs = ef.model_performance_params(y_test_type1, probas_LR[:, 1], model_info['ths_use'], model_info['ths_value'])
    # parameters = model_use.get_params()
    result_each = []
    result_each = ['lr', round(para_LR_bs['auc'],3),
                   round(para_LR_bs['sensitivity'],3), round(para_LR_bs['specificity'],3), round(para_LR_bs['accuracy'],3),
                   round(para_LR_bs['F1'],3), round(para_LR_bs['precision'],3), round(para_LR_bs['ap'],3), 
                   round(para_LR_bs['brier_score'],3), para_LR_bs['threshold']]

    roc_result_need = pd.DataFrame()
    roc_result_need['id'] = data['id_test']
    roc_result_need['true_label'] = y_test_type1
    roc_result_need['lr'] = probas_LR[:, 1]

    # plot figure
    roc_plot_need = pd.DataFrame()
    fpr_LR, tpr_LR, threshold_LR = roc_curve(y_test_type1, probas_LR[:, 1])
    roc_plot_need['fpr'] = fpr_LR
    roc_plot_need['tpr'] = tpr_LR

    # 5. save all needed results
    result_info = {}
    result_info = {'perf_matric':result_each, 'pred_real_info':roc_result_need, 'roc_plot_need':roc_plot_need}
    return result_info


def older_svm_model(data, drop_cols_name, model_info):
    # Support vector machine
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 0. define the variables    
    predicted_svm, probas_svm, para_svm, roc_plot_svm, parameters = [], [], {}, {}, {}
    fpr_svm, tpr_svm, threshold_svm = [], [], []
    X_train_type1, X_cal_type1, X_test_type1, y_train_type1, y_cal_type1, y_test_type1 = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([]), pd.Series([])
    # 1. load the data for usage
    X_train_type1, X_cal_type1, X_test_type1, y_train_type1, y_cal_type1, y_test_type1 = \
    data['X_train'].drop(drop_cols_name, axis=1), data['X_cal'].drop(drop_cols_name, axis=1), \
    data['X_test'].drop(drop_cols_name, axis=1), data['y_train'], data['y_cal'], data['y_test']
    # 2. train model
    if model_info['model_use'] == 'False':
        clf_svm_bs = SVC(kernel='rbf', probability=True)
        clf_svm_bs.fit(X_train_type1, y_train_type1)    
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = clf_svm_bs 
            clf_svm_bs = CalibratedClassifierCV(clf_svm_bs, method='isotonic', cv='prefit')
            clf_svm_bs.fit(X_cal_type1, y_cal_type1)
            model_use = clf_svm_bs
        else:
            model_middle = clf_svm_bs
            model_use = clf_svm_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
    else:
        clf_svm_bs = pickle.load(open(model_info['model_path_full_info'], "rb"))
        if model_info['cal_yn'] == 'True':
            model_middle = clf_svm_bs 
            clf_svm_bs = CalibratedClassifierCV(clf_svm_bs, method='isotonic', cv='prefit')
            clf_svm_bs.fit(X_cal_type1, y_cal_type1)
            model_use = clf_svm_bs
        else:
            model_middle = clf_svm_bs
            model_use = clf_svm_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))        

    # 4. get performance results    
    predicted_svm = model_use.predict(X_test_type1)
    probas_svm = model_use.predict_proba(X_test_type1)
    para_svm_bs, roc_plot_svm_bs = ef.model_performance_params(y_test_type1, probas_svm[:, 1], model_info['ths_use'], model_info['ths_value'])
    # parameters = model_use.get_params()
    result_each = []
    result_each = ['svm', round(para_svm_bs['auc'],3),
                   round(para_svm_bs['sensitivity'],3), round(para_svm_bs['specificity'],3), round(para_svm_bs['accuracy'],3),
                   round(para_svm_bs['F1'],3), round(para_svm_bs['precision'],3), round(para_svm_bs['ap'],3), 
                   round(para_svm_bs['brier_score'],3), para_svm_bs['threshold']]

    roc_result_need = pd.DataFrame()
    roc_result_need['id'] = data['id_test']
    roc_result_need['true_label'] = y_test_type1
    roc_result_need['svm'] = probas_svm[:, 1]

    # plot figure
    roc_plot_need = pd.DataFrame()
    fpr_svm, tpr_svm, threshold_svm = roc_curve(y_test_type1, probas_svm[:, 1])
    roc_plot_need['fpr'] = fpr_svm
    roc_plot_need['tpr'] = tpr_svm
 
    # 5. save all needed results
    result_info = {}
    result_info = {'perf_matric':result_each, 'pred_real_info':roc_result_need, 'roc_plot_need':roc_plot_need}
    return result_info


def older_xgb_model(data, drop_cols_name, model_info):
    """
    :param data: data_type2 dict
    :para model_info: dict of {'model_use':{'existing': , 'params':}, 'model_path_full_info':, 'ths_use':, 'ths_value':, 'cal_yn': , 
    #                          'shap_info':{'shap_yn':, 'image_full_info': , 'image_full_info_bar': , 'fea_table_full_info': },
    #                          'no_cal_model_full_info': , 'cal_model_full_info': }
    # like: '.../xgb_cal.dat', '.../features_ranking.png', '.../features_ranking.csv', '.../no_cal.dat', '.../cal.dat'
    :return data: without considering the group label
    """    
    # XGBoost
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 0. define the variables
    predicted_XG, probas_XG, para_XG, roc_plot_XG, parameters = [], [], {}, {}, {}
    fpr_XG, tpr_XG, threshold_XG = [], [], []
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([]), pd.Series([])
    # 1. load data and generate data for usage
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    data['X_train'].drop(drop_cols_name, axis=1), data['X_cal'].drop(drop_cols_name, axis=1), \
    data['X_test'].drop(drop_cols_name, axis=1), data['y_train'], data['y_cal'], data['y_test']
    X_train_t_XG, X_test_t_XG, y_train_t_XG, y_test_t_XG = train_test_split(X_train_type2, y_train_type2,
                                                                            test_size=0.2, shuffle=True,
                                                                            random_state=0)
    # 2. train or load model                                                                        
    if model_info['model_use']['existing'] == 'False':
        params = []
        params = model_info['model_use']['params']
        clf_XG_bs = xgb.XGBClassifier(**params)
        clf_XG_bs.fit(X_train_t_XG, y_train_t_XG, early_stopping_rounds=80, eval_metric="auc",
                    eval_set=[(X_test_t_XG, y_test_t_XG)])
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = clf_XG_bs
            clf_XG_bs = CalibratedClassifierCV(clf_XG_bs, method='isotonic', cv='prefit')
            clf_XG_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_XG_bs
        else:
            model_middle = clf_XG_bs
            model_use = clf_XG_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # 4. shap explaination if needed
        if model_info['shap_info']['shap_yn'] == 'True':
            explainer = shap.TreeExplainer(model_middle)
            X_train_t = pd.DataFrame(data = X_train_t_XG, columns = X_train_type2.columns.values.tolist())
            shap_values = explainer.shap_values(X_train_t)
            shap.summary_plot(shap_values, X_train_t, show=False)
            plt.savefig(model_info['shap_info']['image_full_info'], dpi=500, bbox_inches='tight')
            plt.close()
            shap.summary_plot(shap_values, X_train_t, plot_type="bar", show=False)
            plt.savefig(model_info['shap_info']['image_full_info_bar'], dpi=500, bbox_inches='tight')
            # '.../features_ranking.png'
            # plt.show()
            plt.close('all')
            vals= np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X_train_t.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            feature_importance.to_csv(model_info['shap_info']['fea_table_full_info'], index=False)
            # '.../features_ranking.csv'

    else:
        clf_XG_bs = pickle.load(open(model_info['model_path_full_info'], "rb"))
        # '.../xgb_cal.dat'
        if model_info['cal_yn'] == 'True':
            model_middle = clf_XG_bs
            clf_XG_bs = CalibratedClassifierCV(clf_XG_bs, method='isotonic', cv='prefit')
            clf_XG_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_XG_bs
        else:
            model_middle = clf_XG_bs
            model_use = clf_XG_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))        
    
    # 5. get performance results
    predicted_XG = model_use.predict(X_test_type2)
    probas_XG = model_use.predict_proba(X_test_type2)
    para_XG_bs, roc_plot_XG = ef.model_performance_params(y_test_type2, probas_XG[:, 1], model_info['ths_use'], model_info['ths_value'])        
    # parameters = model_use.get_params()
    result_each = []
    result_each = ['xgb', round(para_XG_bs['auc'],3),
                   round(para_XG_bs['sensitivity'],3), round(para_XG_bs['specificity'],3), round(para_XG_bs['accuracy'],3),
                   round(para_XG_bs['F1'],3), round(para_XG_bs['precision'],3), round(para_XG_bs['ap'],3), 
                   round(para_XG_bs['brier_score'],3), para_XG_bs['threshold']]

    roc_result_need = pd.DataFrame()
    roc_result_need['id'] = data['id_test']
    roc_result_need['true_label'] = y_test_type2
    roc_result_need['xgb'] = probas_XG[:, 1]

    # plot figure
    roc_plot_need = pd.DataFrame()
    fpr_XG, tpr_XG, threshold_XG = roc_curve(y_test_type2, probas_XG[:, 1])
    roc_plot_need['fpr'] = fpr_XG
    roc_plot_need['tpr'] = tpr_XG

    # 6. save all needed results
    result_info = {}
    result_info = {'perf_matric':result_each, 'pred_real_info':roc_result_need, 'roc_plot_need':roc_plot_need}
    return result_info


def older_rf_model(data, drop_cols_name, model_info):
    # Random Forest
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 0. define the variables    
    predicted_RF, probas_RF, para_RF, roc_plot_RF, parameters = [], [], {}, {}, {}
    fpr_RF, tpr_RF, threshold_RF = [], [], []
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([]), pd.Series([])
    # 1. load the data for usage
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    data['X_train'].drop(drop_cols_name, axis=1), data['X_cal'].drop(drop_cols_name, axis=1), \
    data['X_test'].drop(drop_cols_name, axis=1), data['y_train'], data['y_cal'], data['y_test']
    # 2. train model
    if model_info['model_use'] == 'False':
        clf_RF_bs = RandomForestClassifier(
            max_features='sqrt', n_jobs=-1, oob_score=True, random_state=0,
            max_depth=11, n_estimators=780
        ) # Bayesian tuning acquired
        clf_RF_bs = clf_RF_bs.fit(X_train_type2, y_train_type2)
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = clf_RF_bs    
            clf_RF_bs = CalibratedClassifierCV(clf_RF_bs, method='isotonic', cv='prefit')
            clf_RF_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_RF_bs
        else:
            model_middle = clf_RF_bs
            model_use = clf_RF_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # '.../no_cal.dat', '.../no_cal.dat'
    else:
        clf_RF_bs = pickle.load(open(model_info['model_path_full_info'], "rb"))
        if model_info['cal_yn'] == 'True':
            model_middle = clf_RF_bs    
            clf_RF_bs = CalibratedClassifierCV(clf_RF_bs, method='isotonic', cv='prefit')
            clf_RF_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_RF_bs
        else:
            model_middle = clf_RF_bs
            model_use = clf_RF_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # '.../no_cal.dat', '.../no_cal.dat'

    # 4. get performance results                 
    predicted_RF = model_use.predict(X_test_type2)
    probas_RF = model_use.predict_proba(X_test_type2)
    para_RF_bs, roc_plot_RF_bs = ef.model_performance_params(y_test_type2, probas_RF[:, 1], model_info['ths_use'], model_info['ths_value'])
    # parameters = {}
    # parameters = model_use.get_params()
    result_each = []
    result_each = ['rf', round(para_RF_bs['auc'],3),
                   round(para_RF_bs['sensitivity'],3), round(para_RF_bs['specificity'],3), round(para_RF_bs['accuracy'],3),
                   round(para_RF_bs['F1'],3), round(para_RF_bs['precision'],3), round(para_RF_bs['ap'],3), 
                   round(para_RF_bs['brier_score'],3), para_RF_bs['threshold']]

    roc_result_need = pd.DataFrame()
    roc_result_need['id'] = data['id_test']
    roc_result_need['true_label'] = y_test_type2
    roc_result_need['rf'] = probas_RF[:, 1]

    # plot figure
    roc_plot_need = pd.DataFrame()
    fpr_RF, tpr_RF, threshold_RF = roc_curve(y_test_type2, probas_RF[:, 1])
    roc_plot_need['fpr'] = fpr_RF
    roc_plot_need['tpr'] = tpr_RF

    # 5. save all needed results
    result_info = {}
    result_info = {'perf_matric':result_each, 'pred_real_info':roc_result_need, 'roc_plot_need':roc_plot_need}
    return result_info


def older_nb_model(data, drop_cols_name, model_info):
    # Naive bayesian
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 0. define the variables
    predicted_NB, probas_NB, para_NB, roc_plot_NB, parameters = [], [], {}, {}, {}
    fpr_NB, tpr_NB, threshold_NB = [], [], []
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([]), pd.Series([])
    # 1. load the data for usage
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    data['X_train'].drop(drop_cols_name, axis=1), data['X_cal'].drop(drop_cols_name, axis=1), \
    data['X_test'].drop(drop_cols_name, axis=1), data['y_train'], data['y_cal'], data['y_test']
    # 2. train model
    if model_info['model_use'] == 'False':    
        clf_NB_bs = GaussianNB()
        clf_NB_bs = clf_NB_bs.fit(X_train_type2, y_train_type2)
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = clf_NB_bs    
            clf_NB_bs = CalibratedClassifierCV(clf_NB_bs, method='isotonic', cv='prefit')
            clf_NB_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_NB_bs
        else:
            model_middle = clf_NB_bs
            model_use = clf_NB_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # '.../no_cal.dat', '.../no_cal.dat'
    else:
        clf_NB_bs = pickle.load(open(model_info['model_path_full_info'], "rb"))
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = clf_NB_bs    
            clf_NB_bs = CalibratedClassifierCV(clf_NB_bs, method='isotonic', cv='prefit')
            clf_NB_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_NB_bs
        else:
            model_middle = clf_NB_bs
            model_use = clf_NB_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # '.../no_cal.dat', '.../no_cal.dat'
        
    # 4. get performance results                    
    predicted_NB = model_use.predict(X_test_type2)
    probas_NB = model_use.predict_proba(X_test_type2)
    para_NB_bs, roc_plot_NB_bs = ef.model_performance_params(y_test_type2, probas_NB[:, 1], model_info['ths_use'], model_info['ths_value'])
    # parameters = {}
    # parameters = model_use.get_params()
    result_each = []
    result_each = ['nb', round(para_NB_bs['auc'],3),
                   round(para_NB_bs['sensitivity'],3), round(para_NB_bs['specificity'],3), round(para_NB_bs['accuracy'],3),
                   round(para_NB_bs['F1'],3), round(para_NB_bs['precision'],3), round(para_NB_bs['ap'],3), 
                   round(para_NB_bs['brier_score'],3), para_NB_bs['threshold']]

    roc_result_need = pd.DataFrame()
    roc_result_need['id'] = data['id_test']
    roc_result_need['true_label'] = y_test_type2
    roc_result_need['nb'] = probas_NB[:, 1]

    # plot figure
    roc_plot_need = pd.DataFrame()
    fpr_NB, tpr_NB, threshold_NB = roc_curve(y_test_type2, probas_NB[:, 1])
    roc_plot_need['fpr'] = fpr_NB
    roc_plot_need['tpr'] = tpr_NB

    # 5. save all needed results
    result_info = {}
    result_info = {'perf_matric':result_each, 'pred_real_info':roc_result_need, 'roc_plot_need':roc_plot_need}
    return result_info


def performance_machine_learning_models(data_type1, data_type2, path_save, all_models_info):

    # 1. train and evaluate models
    lr_result_info, svm_result_info, xgb_result_info, rf_result_info, nb_result_info = {}, {}, {}, {}, {}
    drop_cols_name = []
    drop_cols_name = ['apsiii', 'apsiii_prob', 'oasis', 'oasis_prob', 'saps', 'saps_prob', 'sofa', 'sofa_prob']
    lr_result_info = older_lr_model(data_type1, drop_cols_name, all_models_info['lr'])
    svm_result_info = older_svm_model(data_type1, drop_cols_name, all_models_info['svm'])
    xgb_result_info = older_xgb_model(data_type2, drop_cols_name, all_models_info['xgb'])
    rf_result_info = older_rf_model(data_type2, drop_cols_name, all_models_info['rf'])
    nb_result_info = older_nb_model(data_type2, drop_cols_name, all_models_info['nb'])
    
    # 2. save results
    #    2.1 all performance matrics of all models
    data_name = ''
    data_name = 'models_perf_matric.csv'
    perf_matric_all = pd.DataFrame()
    perf_matric_all = pd.DataFrame(list(zip(lr_result_info['perf_matric'], svm_result_info['perf_matric'], \
        xgb_result_info['perf_matric'], rf_result_info['perf_matric'], nb_result_info['perf_matric']))).T
    perf_matric_all.columns = ['name', 'auc', 'sensitivity', 'specificity', 'accuracy', 'F1', \
        'precision', 'ap', 'brier_score', 'threshold']
    perf_matric_all.to_csv(path_save + data_name, index=False)
    #    2.2 all prediction probability info of all models
    data_name = ''
    data_name = 'models_pred_true.csv'
    pred_real_info_all = pd.DataFrame()
    # https://stackoverflow.com/questions/55652704/merge-multiple-dataframes-pandas
    pred_real_info_all_raw = []
    pred_real_info_all_raw = [lr_result_info['pred_real_info'], svm_result_info['pred_real_info'], \
        xgb_result_info['pred_real_info'], rf_result_info['pred_real_info'], nb_result_info['pred_real_info']]
    pred_real_info_all_raw = [df.set_index(['id', 'true_label']) for df in pred_real_info_all_raw]
    pred_real_info_all = pd.concat(pred_real_info_all_raw, axis=1).reset_index()
    pred_real_info_all.to_csv(path_save + data_name, index=False)
    del pred_real_info_all_raw
    #    2.3 all models' roc plot info
    roc_models_set = {'LR':np.stack((lr_result_info['roc_plot_need']['fpr'], lr_result_info['roc_plot_need']['tpr'])).tolist(), \
                      'SVM':np.stack((svm_result_info['roc_plot_need']['fpr'], svm_result_info['roc_plot_need']['tpr'])).tolist(), \
                      'XGB':np.stack((xgb_result_info['roc_plot_need']['fpr'], xgb_result_info['roc_plot_need']['tpr'])).tolist(), \
                      'RF':np.stack((rf_result_info['roc_plot_need']['fpr'], rf_result_info['roc_plot_need']['tpr'])).tolist(), \
                      'NB':np.stack((nb_result_info['roc_plot_need']['fpr'], nb_result_info['roc_plot_need']['tpr'])).tolist()
                      }
   
    return roc_models_set


def performance_scores(data, train_test_name, path_save, all_scores_info):
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 1. get the needed data
    score_name_cal = []
    score_name_cal = ['apsiii', 'sofa', 'oasis', 'saps']
    X_test_score, y_test_score = pd.DataFrame(), pd.Series([])
    X_test_score = data['X_test'][['oasis_prob', 'apsiii_prob', 'saps_prob', 'sofa_prob']]
    dict = {'oasis_prob': 'oasis', 'apsiii_prob': 'apsiii', 'saps_prob': 'saps', 'sofa_prob':'sofa'}
    X_test_score.rename(columns=dict, inplace=True)
    X_test_score['id'] = data['id_test']
    y_test_score = data['y_test']    
    # if test set is 'eicu', we will add apache_iv info
    if train_test_name == 'MIMIC_eICU-eICU':
        apache_iv_info = pd.read_csv('D:/project/older_score/data/' + 'eicu.csv')
        apache_iv_info = apache_iv_info[['id', 'apache_iv_prob']] # 'death_hosp' - for test the right matching
        apache_iv_info.loc[apache_iv_info['apache_iv_prob'] <= 0, 'apache_iv_prob'] = 0.01
        apache_iv_info['apache_iv_prob'].fillna((apache_iv_info['apache_iv_prob'].median()), inplace=True)
        apache_iv_info = apache_iv_info.rename(columns={'apache_iv_prob': 'apache_iv'})        
        X_test_score = X_test_score.merge(apache_iv_info, how='left', on='id')
        score_name_cal.insert(len(score_name_cal), 'apache_iv')

    # 2. get the scores' performance
    #      2.1 all performance matrics of all scores
    data_name = 'scores_perf_matric.csv'
    perf_matric_all = []
    for i in score_name_cal:
        score_parameters = []
        score_parameters = ef.score_performance_cal(X_test_score, y_test_score, i, \
            all_scores_info[train_test_name][i]['ths_use'], all_scores_info[train_test_name][i]['ths_value'])
        perf_matric_all.append(score_parameters)    
    perf_matric_all = pd.DataFrame(perf_matric_all)
    perf_matric_all.columns = ['name', 'auc', 'sensitivity', 'specificity', 'accuracy', 'F1', \
        'precision', 'ap', 'brier_score', 'threshold']
    perf_matric_all.to_csv(path_save + data_name, index=False)
    #    2.2 all prediction probability info of all scores
    data_name = ''
    data_name = 'scores_pred_true.csv'
    pred_real_info_all = pd.DataFrame()
    pred_real_info_all = X_test_score
    pred_real_info_all['true_label'] = y_test_score
    pred_real_info_all.to_csv(path_save + data_name, index=False)
    #    2.3 all scores' roc plot info
    fpr_saps, tpr_saps, threshold_saps = [], [], []
    fpr_sofa, tpr_sofa, threshold_sofa = [], [], []
    fpr_apsiii, tpr_apsiii, threshold_apsiii = [], [], []
    fpr_oasis, tpr_oasis, threshold_oasis = [], [], []
    fpr_saps, tpr_saps, threshold_saps = roc_curve(y_test_score, X_test_score['saps'])  ###计算真正率和假正率
    fpr_sofa, tpr_sofa, threshold_sofa = roc_curve(y_test_score, X_test_score['sofa'])
    fpr_apsiii, tpr_apsiii, threshold_apsiii = roc_curve(y_test_score, X_test_score['apsiii'])
    fpr_oasis, tpr_oasis, threshold_oasis = roc_curve(y_test_score, X_test_score['oasis'])
    if train_test_name == 'MIMIC_eICU-eICU':
        fpr_apache_iv, tpr_apache_iv, threshold_apache_iv = [], [], []
        fpr_apache_iv, tpr_apache_iv, threshold_apache_iv = roc_curve(y_test_score, X_test_score['apache_iv'])    
    
    roc_scores_set = {'saps': np.stack((fpr_saps, tpr_saps)).tolist(), 'oasis': np.stack((fpr_oasis, tpr_oasis)).tolist(),
                      'sofa': np.stack((fpr_sofa, tpr_sofa)).tolist(), 'apsiii': np.stack((fpr_apsiii, tpr_apsiii)).tolist(),
                      }
    if train_test_name == 'MIMIC_eICU-eICU':
        roc_scores_set.update({'apache_iv': np.stack((fpr_apache_iv, tpr_apache_iv)).tolist()})

    return roc_scores_set


def older_xgb_part_fea_model(data, drop_cols_name, model_info):
    """
    :param data: data_type2 dict
    :para model_info: dict of {'model_use':, 'model_path_full_info':, 'ths_use':, 'ths_value':, 'cal_yn': , 
    #                          'shap_info':{'shap_yn':, 'image_full_info': , 'fea_table_full_info': },
    #                          'no_cal_model_full_info': , 'cal_model_full_info': ,
    #                          'feature_rank_list': , 'fea_num_list':}
    # like: '.../xgb_cal.dat', '.../features_ranking.png', '.../features_ranking.csv', '.../no_cal.dat', '.../cal.dat'
    :return data: without considering the group label
    """
    # XGBoost
    # data = {'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test, \
    #         'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test, \
    #         'id_train': id_train, 'id_cal': id_cal, 'id_test': id_test}

    # 0. define the variables
    predicted_XG, probas_XG, para_XG, roc_plot_XG, parameters = [], [], {}, {}, {}
    fpr_XG, tpr_XG, threshold_XG = [], [], []
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series([]), pd.Series([]), pd.Series([])
    # 1. load data and generate data for usage
    X_train_type2, X_cal_type2, X_test_type2, y_train_type2, y_cal_type2, y_test_type2 = \
    data['X_train'].drop(drop_cols_name, axis=1), data['X_cal'].drop(drop_cols_name, axis=1), \
    data['X_test'].drop(drop_cols_name, axis=1), data['y_train'], data['y_cal'], data['y_test']
    X_train_t_XG, X_test_t_XG, y_train_t_XG, y_test_t_XG = train_test_split(X_train_type2, y_train_type2,
                                                                            test_size=0.2, shuffle=True,
                                                                            random_state=0)
    # 2. train or load model                                                                        
    if model_info['model_use'] == 'False':
        
        if model_info['default_use'] == 'False':
            params = []
            params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 
                      'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.025, 'max_delta_step': 0, 'max_depth': 15, 
                      'min_child_weight': 2.0, 'n_estimators': 770, 'n_jobs': -1, 'nthread': None, 
                      'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 
                      'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 0.75, 'verbosity': 1
                      } # miceforest imputation & nonprocess
            clf_XG_bs = xgb.XGBClassifier(**params)
        else:
            clf_XG_bs = xgb.XGBClassifier()

        clf_XG_bs.fit(X_train_t_XG, y_train_t_XG, early_stopping_rounds=80, eval_metric="auc",
                    eval_set=[(X_test_t_XG, y_test_t_XG)])
        # 3. calibrate model if needed
        if model_info['cal_yn'] == 'True':
            model_middle = clf_XG_bs
            clf_XG_bs = CalibratedClassifierCV(clf_XG_bs, method='isotonic', cv='prefit')
            clf_XG_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_XG_bs
        else:
            model_middle = clf_XG_bs
            model_use = clf_XG_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
        # 4. shap explaination if needed
        if model_info['shap_info']['shap_yn'] == 'True':
            explainer = shap.TreeExplainer(model_middle)
            X_train_t = pd.DataFrame(data = X_train_t_XG, columns = X_train_type2.columns.values.tolist())
            shap_values = explainer.shap_values(X_train_t)
            shap.summary_plot(shap_values, X_train_t, show=False)
            plt.savefig(model_info['shap_info']['image_full_info'], dpi=500, bbox_inches='tight')
            # '.../features_ranking.png'
            # plt.show()
            plt.close('all')
            vals= np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X_train_t.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            feature_importance.to_csv(model_info['shap_info']['fea_table_full_info'], index=False)
            # '.../features_ranking.csv'

    else:
        clf_XG_bs = pickle.load(open(model_info['model_path_full_info'], "rb"))
        # '.../xgb_cal.dat'
        if model_info['cal_yn'] == 'True':
            model_middle = clf_XG_bs
            clf_XG_bs = CalibratedClassifierCV(clf_XG_bs, method='isotonic', cv='prefit')
            clf_XG_bs.fit(X_cal_type2, y_cal_type2)
            model_use = clf_XG_bs
        else:
            model_middle = clf_XG_bs
            model_use = clf_XG_bs
        # save model
        pickle.dump(model_middle, open(model_info['no_cal_model_full_info'], "wb"))
        pickle.dump(model_use, open(model_info['cal_model_full_info'], "wb"))
    
    # 5. get performance results
    predicted_XG = model_use.predict(X_test_type2)
    probas_XG = model_use.predict_proba(X_test_type2)
    para_XG_bs, roc_plot_XG = ef.model_performance_params(y_test_type2, probas_XG[:, 1], model_info['ths_use'], model_info['ths_value'])        
    # parameters = model_use.get_params()
    result_each = []
    result_each = ['xgb', round(para_XG_bs['auc'],3),
                   round(para_XG_bs['sensitivity'],3), round(para_XG_bs['specificity'],3), round(para_XG_bs['accuracy'],3),
                   round(para_XG_bs['F1'],3), round(para_XG_bs['precision'],3), round(para_XG_bs['ap'],3), 
                   round(para_XG_bs['brier_score'],3), para_XG_bs['threshold']]

    roc_result_need = pd.DataFrame()
    roc_result_need['id'] = data['id_test']
    roc_result_need['true_label'] = y_test_type2
    roc_result_need['xgb'] = probas_XG[:, 1]

    # plot figure
    roc_plot_need = pd.DataFrame()
    fpr_XG, tpr_XG, threshold_XG = roc_curve(y_test_type2, probas_XG[:, 1])
    roc_plot_need['fpr'] = fpr_XG
    roc_plot_need['tpr'] = tpr_XG

    # 6. save all needed results
    result_info = {}
    result_info = {'perf_matric':result_each, 'pred_real_info':roc_result_need, 'roc_plot_need':roc_plot_need}
    return result_info