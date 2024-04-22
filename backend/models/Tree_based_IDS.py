
## Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance


def tree_based_IDS(file):

    metrics = {}
    
    # # Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles 
    # This is the code for the paper entitled "[**Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles**](https://arxiv.org/pdf/1910.08635.pdf)" published in IEEE GlobeCom 2019.  
    # Authors: Li Yang (liyanghart@gmail.com), Abdallah Moubayed, Ismail Hamieh, and Abdallah Shami  
    # Organization: The Optimized Computing and Communications (OC2) Lab, ECE Department, Western University

    # If you find this repository useful in your research, please cite:  
    # L. Yang, A. Moubayed, I. Hamieh and A. Shami, "Tree-Based Intelligent Intrusion Detection System in Internet of Vehicles," 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013892.  

    ### Preprocessing (normalization and padding values)
    df = pd.read_csv(f'{file}.csv')

    # Min-max normalization
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))
    # Fill empty values by 0
    df = df.fillna(0)

    ### split train set and test set
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop(['Label'],axis=1).values 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

    # X_train.shape

    # pd.Series(y_train).value_counts()

    ### Oversampling by SMOTE

    from imblearn.over_sampling import SMOTE
    y_train_encoded = labelencoder.fit_transform(y_train)
    smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500}) # Create 1500 samples for the minority class "4"

    X_train, y_train = smote.fit_resample(X_train, y_train_encoded)

    # pd.Series(y_train).value_counts()

    ## Machine learning model training

    ### Training four base learners: decision tree, random forest, extra trees, XGBoost

    # Decision tree training and prediction
    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(X_train,y_train) 
    y_test_encoded = labelencoder.fit_transform(y_test)
    dt_score=dt.score(X_test,y_test_encoded)
    y_predict=dt.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))

    metrics.update([('Decision Tree', {
        "Accuracy" : str(dt_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)

    # Random Forest training and prediction
    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) 
    rf_score=rf.score(X_test,y_test_encoded)
    y_predict=rf.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    #print(classification_report(y_true,y_predict))
    
    metrics.update([('Random Forest', {
        "Accuracy" : str(rf_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    #cm=confusion_matrix(y_true,y_predict)

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)

    # Extra trees training and prediction
    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test_encoded)
    y_predict=et.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('Extra Trees', {
        "Accuracy" : str(et_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    et_train=et.predict(X_train)
    et_test=et.predict(X_test)

    # XGboost training and prediction
    xg = xgb.XGBClassifier(n_estimators = 10)
    xg.fit(X_train,y_train)
    xg_score=xg.score(X_test,y_test_encoded)
    y_predict=xg.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('XGBoost', {
        "Accuracy" : str(xg_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)

    ### Stacking model construction (ensemble for 4 base learners)

    # Use the outputs of 4 base models to construct a new ensemble model
    base_predictions_train = pd.DataFrame( {
        'DecisionTree': dt_train.ravel(),
            'RandomForest': rf_train.ravel(),
        'ExtraTrees': et_train.ravel(),
        'XgBoost': xg_train.ravel(),
        })

    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    xg_train=xg_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)
    xg_test=xg_test.reshape(-1, 1)

    x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)

    stk = xgb.XGBClassifier().fit(x_train, y_train)

    y_predict=stk.predict(x_test)
    y_true=y_test_encoded
    stk_score=accuracy_score(y_true,y_predict)
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('Stacking', {
        "Accuracy" : str(stk_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])    
    # cm=confusion_matrix(y_true,y_predict)

    ## Feature Selection

    ### Feature importance

    # Save the feature importance lists generated by four tree-based algorithms
    dt_feature = dt.feature_importances_
    rf_feature = rf.feature_importances_
    et_feature = et.feature_importances_
    xgb_feature = xg.feature_importances_

    # calculate the average importance value of each feature
    avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature)/4

    feature=(df.drop(['Label'],axis=1)).columns.values
    # print ("Features sorted by their score:")
    # print (sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True))

    f_list = sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)

    # len(f_list)

    # Select the important features from top-importance to bottom-importance until the accumulated importance reaches 0.9 (out of 1)
    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])
        if Sum>=0.9:
            break        

    X_fs = df[fs].values

    X_train, X_test, y_train, y_test = train_test_split(X_fs,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

    # X_train.shape

    # pd.Series(y_train).value_counts()

    ### Oversampling by SMOTE

    from imblearn.over_sampling import SMOTE
    smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500})

    X_train, y_train = smote.fit_resample(X_train, y_train_encoded)

    # pd.Series(y_train).value_counts()

    ## Machine learning model training after feature selection

    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(X_train,y_train) 
    dt_score=dt.score(X_test,y_test_encoded)
    y_predict=dt.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('(FS) Decision Tree', {
        "Accuracy" : str(dt_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)

    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) # modelin veri üzerinde öğrenmesi fit fonksiyonuyla yapılıyor
    rf_score=rf.score(X_test,y_test_encoded)
    y_predict=rf.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
    
    metrics.update([('(FS) Random Forest', {
        "Accuracy" : str(rf_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])

    # cm=confusion_matrix(y_true,y_predict)

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)

    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test_encoded)
    y_predict=et.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('(FS) Extra Trees', {
        "Accuracy" : str(et_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    et_train=et.predict(X_train)
    et_test=et.predict(X_test)

    xg = xgb.XGBClassifier(n_estimators = 10)
    xg.fit(X_train,y_train)
    xg_score=xg.score(X_test,y_test_encoded)
    y_predict=xg.predict(X_test)
    y_true=y_test_encoded
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('(FS) XGBoost', {
        "Accuracy" : str(xg_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    xg_train=xg.predict(X_train)
    xg_test=xg.predict(X_test)

    ### Stacking model construction

    base_predictions_train = pd.DataFrame( {
        'DecisionTree': dt_train.ravel(),
            'RandomForest': rf_train.ravel(),
        'ExtraTrees': et_train.ravel(),
        'XgBoost': xg_train.ravel(),
        })

    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    xg_train=xg_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)
    xg_test=xg_test.reshape(-1, 1)

    x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)

    stk = xgb.XGBClassifier().fit(x_train, y_train)
    y_predict=stk.predict(x_test)
    y_true=y_test_encoded
    stk_score=accuracy_score(y_true,y_predict)
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    # print(classification_report(y_true,y_predict))
    
    metrics.update([('(FS) Stacking', {
        "Accuracy" : str(stk_score*100),
        "Precision" : str(precision*100),
        "Recall" : str(recall*100),
        "F1_score" : str(fscore*100)
    })])
    # cm=confusion_matrix(y_true,y_predict)

    return metrics




