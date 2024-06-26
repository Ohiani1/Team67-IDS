import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
from statistics import mode
from .lccde import LCCDE

def LCCDE_IDS(file):

    metrics = {}
    df = pd.read_csv(f"{file}.csv")

    # Replace infinity values with NaN
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)

    # Replace extremely large values with NaN
    max_value = 1e+15  # Set the maximum allowed value
    df[numerical_cols] = df[numerical_cols].mask(df[numerical_cols].abs() > max_value, np.nan)

    df = df.dropna()

    # Create a dictionary to map labels to their numeric values
    label_map = {'BENIGN': 0, 'Bot': 1, 'BruteForce': 2, 'DoS': 3, 'Infiltration': 4, 'PortScan': 5, 'WebAttack': 6}

    # Replace the non-numeric labels with their numeric values
    df['Label'] = df['Label'].apply(lambda x: label_map.get(x, x) if pd.notna(x) and not str(x).isdigit() else x)

    X = df.drop(['Label'],axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0) #shuffle=False

    from imblearn.over_sampling import SMOTE
    from collections import Counter
    # smote=SMOTE(n_jobs=-1,sampling_strategy={2:1000,4:1000})

    # Get the class distribution
    class_counts = Counter(y)
    print("Class distribution:", class_counts)

    # Define the sampling_strategy
    sampling_strategy = {}
    for class_label, count in class_counts.items():
        if count < 1000:  # Adjust the desired number of samples as per your requirements
            sampling_strategy[class_label] = 1000  # Oversample minority classes to 1000 samples

    # Apply SMOTE with the defined sampling_strategy
    smote = SMOTE(n_jobs=-1, sampling_strategy=sampling_strategy)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train the LightGBM algorithm
    import lightgbm as lgb
    lg = lgb.LGBMClassifier()
    lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)
    # print(classification_report(y_test,y_pred))

    metrics.update([('LightGBM',{
        "Accuracy": str(accuracy_score(y_test, y_pred)*100),
        "Precision": str(precision_score(y_test, y_pred, average='weighted')*100),
        "Recall":str(recall_score(y_test, y_pred, average='weighted')*100),
        "F1_score": str(f1_score(y_test, y_pred, average='weighted')*100),
        "F1_or_each_type_of_attack": str(f1_score(y_test, y_pred, average=None))
    })])
    lg_f1=f1_score(y_test, y_pred, average=None)

    # Train the XGBoost algorithm
    import xgboost as xgb
    xg = xgb.XGBClassifier()

    X_train_x = X_train.values
    X_test_x = X_test.values

    xg.fit(X_train_x, y_train)

    y_pred = xg.predict(X_test_x)
    # print(classification_report(y_test,y_pred))


    metrics.update([('XGBoost',{
        "Accuracy": str(accuracy_score(y_test, y_pred)*100),
        "Precision": str(precision_score(y_test, y_pred, average='weighted')*100),
        "Recall":str(recall_score(y_test, y_pred, average='weighted')*100),
        "F1_score": str(f1_score(y_test, y_pred, average='weighted')*100),
        "F1_or_each_type_of_attack": str(f1_score(y_test, y_pred, average=None))
    })])
    xg_f1=f1_score(y_test, y_pred, average=None)

    # Train the CatBoost algorithm
    import catboost as cbt
    cb = cbt.CatBoostClassifier(verbose=0,boosting_type='Plain')
    #cb = cbt.CatBoostClassifier()

    cb.fit(X_train, y_train)
    y_pred = cb.predict(X_test)
    # print(classification_report(y_test,y_pred))

    metrics.update([('CatBoost',{
        "Accuracy": str(round(accuracy_score(y_test, y_pred)*100,2)),
        "Precision": str(precision_score(y_test, y_pred, average='weighted')*100),
        "Recall":str(recall_score(y_test, y_pred, average='weighted')*100),
        "F1_score": str(f1_score(y_test, y_pred, average='weighted')*100),
        "F1_or_each_type_of_attack": str(f1_score(y_test, y_pred, average=None))
    })])
    cb_f1=f1_score(y_test, y_pred, average=None)

    model=[]
    for i in range(len(lg_f1)):
        if max(lg_f1[i],xg_f1[i],cb_f1[i]) == lg_f1[i]:
            model.append(lg)
        elif max(lg_f1[i],xg_f1[i],cb_f1[i]) == xg_f1[i]:
            model.append(xg)
        else:
            model.append(cb)

    # Implementing LCCDE
    yt, yp = LCCDE(X_test, y_test, m1 = lg, m2 = xg, m3 = cb, model = model)

    # The performance of the proposed lCCDE model
    metrics.update([('LCCDE',{
        "Accuracy": str(accuracy_score(yt, yp)*100),
        "Precision": str(precision_score(yt, yp, average='weighted')*100),
        "Recall":str(recall_score(yt, yp, average='weighted')*100),
        "F1_score": str(f1_score(yt, yp, average='weighted')*100),
        "F1_or_each_type_of_attack": str(f1_score(yt, yp, average=None))
    })])

    return metrics
