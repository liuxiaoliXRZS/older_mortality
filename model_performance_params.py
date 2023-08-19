# a function to calculate matrix of model
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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
                  'F1': F1, 'precision': precision_c, 'ap':average_precision, 'threshold': threshold_final}
    roc_plot_data = {'fpr_data': fpr, 'tpr_data': tpr}
    return parameters, roc_plot_data
