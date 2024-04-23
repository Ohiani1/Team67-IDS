#!/usr/bin/env python
# coding: utf-8

# # MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles
# This is the code for the paper entitled "[**MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles**](https://arxiv.org/pdf/2105.13289.pdf)" accepted in IEEE Internet of Things Journal.  
# Authors: Li Yang (liyanghart@gmail.com), Abdallah Moubayed, and Abdallah Shami  
# Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University
# 
# If you find this repository useful in your research, please cite:  
# L. Yang, A. Moubayed, and A. Shami, “MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles,” IEEE Internet of Things Journal, vol. 9, no. 1, pp. 616-632, Jan.1, 2022.

# ## Import libraries


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance




def MHT_IDS(file, lr=None, ne=None, md=None):
    metrics = {}
    df = pd.read_csv('CICIDS2017_sample.csv')
    # The results in this code is based on the original CICIDS2017 dataset.

    label_map = {'BENIGN': 0, 'Bot': 1, 'BruteForce': 2, 'DoS': 3, 'Infiltration': 4, 'PortScan': 5, 'WebAttack': 6}

    # Replace the non-numeric labels with their numeric values
    df['Label'] = df['Label'].apply(lambda x: label_map.get(x, x) if pd.notna(x) and not str(x).isdigit() else x)
    df['Label'] = df['Label'].astype(int).astype(str)

    # ### Preprocessing (normalization and padding values)

    # Z-score normalization
    features = df.dtypes[df.dtypes != 'object'].index
    df[features] = df[features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Fill empty values by 0
    df = df.fillna(0)

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

    # retain the minority class instances and sample the majority class instances
    df_minor = df[(df['Label'] == 6) | (df['Label'] == 1) | (df['Label'] == 4)]
    df_major = df.drop(df_minor.index)

    X = df_major.drop(['Label'], axis=1)
    y = df_major.iloc[:, -1].values.reshape(-1, 1)
    y = np.ravel(y)

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)

    klabel = kmeans.labels_
    df_major['klabel'] = klabel

    cols = list(df_major)
    cols.insert(78, cols.pop(cols.index('Label')))
    df_major = df_major.loc[:, cols]

    def typicalSampling(group):
        name = group.name
        frac = 0.008
        return group.sample(frac=frac)

    result = df_major.groupby(
        'klabel', group_keys=False
    ).apply(typicalSampling)

    result = result.drop(['klabel'], axis=1)
    result = pd.concat([result, df_minor])

    # ### split train set and test set

    # Read the sampled dataset
    df = pd.read_csv(f'{file}.csv')

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)

    # Replace extremely large values with NaN
    max_value = 1e+15  # Set the maximum allowed value
    df[numerical_cols] = df[numerical_cols].mask(df[numerical_cols].abs() > max_value, np.nan)
    df = df.dropna(how='any')
    label_map = {'BENIGN': 0, 'Bot': 1, 'BruteForce': 2, 'DoS': 3, 'Infiltration': 4, 'PortScan': 5, 'WebAttack': 6}
    df['Label'] = df['Label'].apply(lambda x: label_map.get(x, x) if pd.notna(x) and not str(x).isdigit() else x)

    X = df.drop(['Label'], axis=1).values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

    from sklearn.feature_selection import mutual_info_classif
    importances = mutual_info_classif(X_train, y_train)

    # calculate the sum of importance scores
    f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])

    # select the important features from top to bottom until the accumulated importance reaches 90%
    f_list2 = sorted(zip(map(lambda x: round(x, 4), importances / Sum), features), reverse=True)
    Sum2 = 0
    fs = []
    for i in range(0, len(f_list2)):
        Sum2 = Sum2 + f_list2[i][0]
        fs.append(f_list2[i][1])
        if Sum2 >= 0.9:
            break

    X_fs = df[fs].values

    from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
    fcbf = FCBFK(k=20)
    # fcbf.fit(X_fs, y)

    X_fss = fcbf.fit_transform(X_fs, y)

    # ### Re-split train & test sets after feature selection

    X_train, X_test, y_train, y_test = train_test_split(X_fss, y, train_size=0.8, test_size=0.2, random_state=0,
                                                        stratify=y)

    # ### SMOTE to solve class-imbalance

    from imblearn.over_sampling import SMOTE
    from collections import Counter

    sampling_strategy = {}

    class_counts = Counter(y)

    for class_label, count in class_counts.items():
        if count < 1000:  # Adjust the desired number of samples as per your requirements
            sampling_strategy[class_label] = 1000  # Oversample minority classes to 1000 samples

    smote = SMOTE(n_jobs=-1, sampling_strategy=sampling_strategy)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ## Machine learning model training

    # ### Training four base learners: decision tree, random forest, extra trees, XGBoost

    # #### Apply XGBoost

    xg = xgb.XGBClassifier(n_estimators=10)
    xg.fit(X_train, y_train)
    xg_score = xg.score(X_test, y_test)
    y_predict = xg.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    xg = xgb.XGBClassifier(learning_rate=0.7340229699980686, n_estimators=70, max_depth=14)
    xg.fit(X_train, y_train)
    xg_score = xg.score(X_test, y_test)
    y_predict = xg.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    xg_train = xg.predict(X_train)
    xg_test = xg.predict(X_test)

    # #### Apply RF

    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    y_predict = rf.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    # #### Hyperparameter optimization (HPO) of random forest using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
    # Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

    rf_hpo = RandomForestClassifier(n_estimators=71, min_samples_leaf=1, max_depth=46, min_samples_split=9,
                                    max_features=20, criterion='entropy')
    rf_hpo.fit(X_train, y_train)
    rf_score = rf_hpo.score(X_test, y_test)
    y_predict = rf_hpo.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    rf_train = rf_hpo.predict(X_train)
    rf_test = rf_hpo.predict(X_test)

    # #### Apply DT

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    dt_score = dt.score(X_test, y_test)
    y_predict = dt.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    # #### Hyperparameter optimization (HPO) of decision tree using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
    # Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

    dt_hpo = DecisionTreeClassifier(min_samples_leaf=2, max_depth=47, min_samples_split=3, max_features=19,
                                    criterion='gini')
    dt_hpo.fit(X_train, y_train)
    dt_score = dt_hpo.score(X_test, y_test)
    y_predict = dt_hpo.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    dt_train = dt_hpo.predict(X_train)
    dt_test = dt_hpo.predict(X_test)

    # #### Apply ET

    et = ExtraTreesClassifier(random_state=0)
    et.fit(X_train, y_train)
    et_score = et.score(X_test, y_test)
    y_predict = et.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    # #### Hyperparameter optimization (HPO) of extra trees using Bayesian optimization with tree-based Parzen estimator (BO-TPE)
    # Based on the GitHub repo for HPO: https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms

    et_hpo = ExtraTreesClassifier(n_estimators=53, min_samples_leaf=1, max_depth=31, min_samples_split=5,
                                  max_features=20, criterion='entropy')
    et_hpo.fit(X_train, y_train)
    et_score = et_hpo.score(X_test, y_test)
    y_predict = et_hpo.predict(X_test)
    y_true = y_test
    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    et_train = et_hpo.predict(X_train)
    et_test = et_hpo.predict(X_test)

    # ### Apply Stacking
    # The ensemble model that combines the four ML models (DT, RF, ET, XGBoost)

    base_predictions_train = pd.DataFrame({
        'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
        'ExtraTrees': et_train.ravel(),
        'XgBoost': xg_train.ravel(),
    })

    dt_train = dt_train.reshape(-1, 1)
    et_train = et_train.reshape(-1, 1)
    rf_train = rf_train.reshape(-1, 1)
    xg_train = xg_train.reshape(-1, 1)
    dt_test = dt_test.reshape(-1, 1)
    et_test = et_test.reshape(-1, 1)
    rf_test = rf_test.reshape(-1, 1)
    xg_test = xg_test.reshape(-1, 1)

    x_train = np.concatenate((dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate((dt_test, et_test, rf_test, xg_test), axis=1)

    stk = xgb.XGBClassifier(
        learning_rate=lr if lr is not None else 0.19229249758051492,
        n_estimators=ne if ne is not None else 30,
        max_depth=md if md is not None else 36
    ).fit(x_train, y_train)
    y_predict = stk.predict(x_test)
    y_true = y_test
    stk_score = accuracy_score(y_true, y_predict)

    precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')

    metrics.update([('MHT', {
        "Accuracy": str(stk_score * 100),
        "Precision": str(precision * 100),
        "Recall": str(recall * 100),
        "F1_score": str(fscore * 100),
    })])

    return metrics





