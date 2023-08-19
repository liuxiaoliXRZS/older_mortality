#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, STATUS_OK, tpe, space_eval, Trials
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import model_performance_params as mpp


"""
XGBoost
"""


def train_xgb_model(x_train, y_train, x_eval, y_eval, x_test, y_test, iteration_num, scale_pos_weight_use):
    trials = Trials()

    def xgb_obj_func(params):
        clf = xgb.XGBClassifier(**params, n_jobs=-1, tree_method='approx')
        clf_new = clf.fit(x_train, y_train, early_stopping_rounds=80, eval_metric="auc", eval_set=[(x_eval, y_eval)])
        predicted_clf = clf_new.predict(x_test)
        probas_clf = clf_new.predict_proba(x_test)
        para_clf, roc_plot_clf = mpp.model_performance_params(y_test.values, probas_clf[:, 1], predicted_clf)
        return {'loss': 1 - para_clf['auc'], 'status': STATUS_OK}

    xgb_space = {
        'objective': 'binary:logistic',
        'metric': 'auc',
        'scale_pos_weight': scale_pos_weight_use,
        'learning_rate': hp.quniform('learning_rate', 0.025, 0.3, 0.025),
        'n_estimators': hp.choice('n_estimators', np.arange(200, 1000, 10, dtype=int)),
        'max_depth': hp.choice('max_depth', np.arange(4, 16, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05)
    }

    best_xgb = fmin(fn=xgb_obj_func,
                    space=xgb_space,
                    algo=tpe.suggest,
                    show_progressbar=False,
                    max_evals=iteration_num,
                    trials=trials,
                    rstate=np.random.RandomState(123)
                    )

    best_params = space_eval(xgb_space, best_xgb)

    best_xgb_clf = xgb.XGBClassifier(max_depth=best_params['max_depth'],
                                     learning_rate=best_params['learning_rate'],
                                     n_estimators=best_params['n_estimators'],
                                     subsample=best_params['subsample'],
                                     min_child_weight=best_params['min_child_weight'],
                                     scale_pos_weight=scale_pos_weight_use,
                                     n_jobs=-1
                                     )

    best_xgb_clf.fit(
        x_train, y_train, eval_set=[(x_eval, y_eval)],
        eval_metric=['auc'], verbose=False, early_stopping_rounds=80
    )

    predicted_clf = best_xgb_clf.predict(x_test)
    probas_clf = best_xgb_clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test.values, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, best_xgb_clf.get_params()


"""
XGBoost Baseline model
"""


def train_xgb_base_model(x_train, y_train, x_eval, y_eval, x_test, y_test):

    clf = xgb.XGBClassifier(
        n_jobs=-1, objective='binary:logistic',
        eval_metric='auc', silent=1, tree_method='approx'
    )
    clf.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(x_eval, y_eval)])
    predicted_clf = clf.predict(x_test)
    probas_clf = clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, clf.get_params()


"""
Random Forest
"""


def train_rf_model(x_train, y_train, x_eval, y_eval, x_test, y_test):
    trials = Trials()

    def rf_obj_func(params):
        clf = RandomForestClassifier(**params, n_jobs=-1, oob_score=True, random_state=0)
        clf_new = clf.fit(x_train, y_train)
        predicted_clf = clf_new.predict(x_eval)
        probas_clf = clf_new.predict_proba(x_eval)
        para_clf, roc_plot_clf = mpp.model_performance_params(y_eval.values, probas_clf[:, 1], predicted_clf)
        return {'loss': 1 - para_clf['auc'], 'status': STATUS_OK}

    rf_space = {'max_depth': hp.choice('max_depth', np.arange(3, 16, dtype=int)),
                'max_features': hp.choice('max_features', np.arange(1, x_train.shape[1], dtype=int)),
                'n_estimators': hp.choice('n_estimators', np.arange(200, 1000, 10, dtype=int))
                }

    best_rf = fmin(fn=rf_obj_func,
                   space=rf_space,
                   algo=tpe.suggest,
                   show_progressbar=False,
                   max_evals=3,
                   trials=trials,
                   rstate=np.random.RandomState(123)
                   )

    best_params = space_eval(rf_space, best_rf)

    best_rf_clf = RandomForestClassifier(max_depth=best_params['max_depth'],
                                         max_features=best_params['max_depth'],
                                         n_estimators=best_params['n_estimators'],
                                         n_jobs=-1, oob_score=True, random_state=0
                                         )

    best_rf_clf.fit(x_train, y_train)
    predicted_clf = best_rf_clf.predict(x_test)
    probas_clf = best_rf_clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test.values, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, best_rf_clf.get_params()


"""
Random Forest baseline model
"""


def train_rf_base_model(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_features='sqrt', n_jobs=-1, oob_score=True, random_state=0)
    clf_new = clf.fit(x_train, y_train)
    predicted_clf = clf_new.predict(x_test)
    probas_clf = clf_new.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, clf_new.get_params()


"""
Support Vector Machine
"""


def train_svm_model(x_train, y_train, x_eval, y_eval, x_test, y_test):
    trials = Trials()

    def svm_obj_func(params):
        clf = SVC(**params, kernel='rbf', probability=True)
        clf_new = clf.fit(x_train, y_train)
        predicted_clf = clf_new.predict(x_eval)
        probas_clf = clf_new.predict_proba(x_eval)
        para_clf, roc_plot_clf = mpp.model_performance_params(y_eval.values, probas_clf[:, 1], predicted_clf)
        return {'loss': 1 - para_clf['auc'], 'status': STATUS_OK}

    svm_space = {
        'C': hp.loguniform('C', -4.0 * np.log(10.0), 4.0 * np.log(10.0)),
    }

    best_svm = fmin(fn=svm_obj_func,
                    space=svm_space,
                    algo=tpe.suggest,
                    show_progressbar=False,
                    max_evals=50,
                    trials=trials,
                    rstate=np.random.RandomState(123)
                    )

    best_params = space_eval(svm_space, best_svm)

    best_svm_clf = SVC(C=best_params['C'], kernel='rbf', probability=True)

    best_svm_clf.fit(x_train, y_train)
    predicted_clf = best_svm_clf.predict(x_test)
    probas_clf = best_svm_clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test.values, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, best_svm_clf.get_params()


"""
Support Vector Machine baseline
"""


def train_svm_base_model(x_train, y_train, x_test, y_test):
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(x_train, y_train)
    predicted_clf = clf.predict(x_test)
    probas_clf = clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, clf.get_params()


"""
Logistic Regression Model
"""


def train_lr_model(x_train, y_train, x_eval, y_eval, x_test, y_test):
    trials = Trials()

    def lr_obj_func(params):
        clf = LogisticRegression(**params, multi_class="ovr", penalty="l1", class_weight="balanced", solver="liblinear")
        clf_new = clf.fit(x_train, y_train)
        predicted_clf = clf_new.predict(x_eval)
        probas_clf = clf_new.predict_proba(x_eval)
        para_clf, roc_plot_clf = mpp.model_performance_params(y_eval.values, probas_clf[:, 1], predicted_clf)
        return {'loss': 1 - para_clf['auc'], 'status': STATUS_OK}

    lr_space = {
        'C': hp.loguniform('C', -4.0 * np.log(10.0), 4.0 * np.log(10.0))
    }

    best_lr = fmin(fn=lr_obj_func,
                   space=lr_space,
                   algo=tpe.suggest,
                   show_progressbar=False,
                   max_evals=5,
                   trials=trials,
                   rstate=np.random.RandomState(123)
                   )

    best_params = space_eval(lr_space, best_lr)

    best_lr_clf = LogisticRegression(C=best_params['C'], multi_class="ovr", penalty="l1", class_weight="balanced", solver="liblinear")

    best_lr_clf.fit(x_train, y_train)
    predicted_clf = best_lr_clf.predict(x_test)
    probas_clf = best_lr_clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test.values, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, best_lr_clf.get_params()


"""
Logistic regression model baseline
"""


def train_lr_base_model(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(multi_class="ovr", penalty="l1", class_weight="balanced", solver="liblinear")
    clf_new = clf.fit(x_train, y_train)
    predicted_clf = clf_new.predict(x_test)
    probas_clf = clf_new.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, clf_new.get_params()


"""
Neural Network
"""


def train_nn_model(x_train, y_train, x_eval, y_eval, x_test, y_test):
    trials = Trials()

    def nn_obj_func(params):
        clf = MLPClassifier(**params)
        clf_new = clf.fit(x_train, y_train)
        predicted_clf = clf_new.predict(x_eval)
        probas_clf = clf_new.predict_proba(x_eval)
        para_clf, roc_plot_clf = mpp.model_performance_params(y_eval.values, probas_clf[:, 1], predicted_clf)
        return {'loss': 1 - para_clf['auc'], 'status': STATUS_OK}

    nn_space = {
        'hidden_layer_sizes': 5 + hp.randint('hidden_layer_sizes', 45),
        'alpha': hp.uniform('alpha', 0.0001, 0.05),
        'activation': hp.choice('activation', ['tanh', 'relu']),
        'solver': hp.choice('solver', ['sgd', 'adam'])
    }

    best_nn = fmin(fn=nn_obj_func,
                   space=nn_space,
                   algo=tpe.suggest,
                   show_progressbar=False,
                   max_evals=50,
                   trials=trials,
                   rstate=np.random.RandomState(123)
                   )

    best_params = space_eval(nn_space, best_nn)

    best_nn_clf = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        alpha=best_params['alpha'],
        activation=best_params['activation'],
        solver=best_params['solver']
    )

    best_nn_clf.fit(x_train, y_train)
    predicted_clf = best_nn_clf.predict(x_test)
    probas_clf = best_nn_clf.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test.values, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, best_nn_clf.get_params()


"""
Neural network baseline model
"""


def train_nn_base_model(x_train, y_train, x_test, y_test):
    clf = MLPClassifier()
    clf_new = clf.fit(x_train, y_train)
    predicted_clf = clf_new.predict(x_test)
    probas_clf = clf_new.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, clf_new.get_params()


"""
Naive bayes
"""


def train_nb_base_model(x_train, y_train, x_test, y_test):
    clf = GaussianNB()
    clf_new = clf.fit(x_train, y_train)
    predicted_clf = clf_new.predict(x_test)
    probas_clf = clf_new.predict_proba(x_test)
    para_clf, roc_plot_clf = mpp.model_performance_params(y_test, probas_clf[:, 1], predicted_clf)

    return para_clf, roc_plot_clf, clf_new.get_params()
