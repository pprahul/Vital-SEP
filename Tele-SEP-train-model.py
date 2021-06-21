import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

VERBOSE = 0
QUANTIZE = 0

time_window_hr = 3
prediction_time_hr = 5
nonsepsis_start_time_hr = 0

monitoring_window_start = nonsepsis_start_time_hr
monitoring_window_end = nonsepsis_start_time_hr + time_window_hr

nonsepsis_dir='CinC2019 Dataset/training_setA/trainingA_nonsepsis'

nonsepsis = []
features = pd.DataFrame()
count = 1

#browsing through non-sepsis data 
for filename in os.listdir(nonsepsis_dir):
        
    # read file as pandas dataframe
    df = pd.read_csv(nonsepsis_dir+'/'+filename,sep='|')
    
    if df.index[-1] >= 15:
        print(count)
        #df_features = df[['HR','O2Sat','SBP','MAP','DBP','Resp','SepsisLabel']]
        df_features = df[['O2Sat','RR']]

        df_features_12h = df_features.iloc[monitoring_window_start:monitoring_window_end,:]
        #print(df_features_12h)
        df_features_12h.fillna(method='ffill', inplace=True, limit=2) #replacing Nan with previous value
        df_features_12h.fillna(method='bfill', inplace=True, limit=2) #replacing Nan with next value (if first row is Nan)
        #print(df_features_12h)
        if (df_features_12h.isnull().values.any() == True):
            continue
        
        if QUANTIZE == 1:
            #Quantizing according to NEWS2
            df_features_12h = quantize_news2(df_features_12h)
        
        
        # Feature construction
        features = pd.DataFrame(columns=df_features_12h.columns)
        features = features.append(df_features_12h.iloc[0])
        
        if (df_features_12h.shape[0] > 1):
            i = 1
            
            while(i < df_features_12h.shape[0]): # find delt from first observation row and rate of change
                delta_from_first_obs = features.iloc[0] - df_features_12h.iloc[i]
                rate_delta_from_first_obs = delta_from_first_obs / i # rate of change
                
                features = features.append(delta_from_first_obs, ignore_index=True)
                features = features.append(rate_delta_from_first_obs, ignore_index=True)
                i+=1
            
            i=2
            while(i < df_features_12h.shape[0]):  # find delt between consecutive observations
                delta_consecutive = df_features_12h.iloc[i-1] - df_features_12h.iloc[i]
                #print(delta_consecutive)
                features = features.append(delta_consecutive, ignore_index=True)
                i+=1
                
        features = features.append(df_features_12h.var(axis=0), ignore_index=True)
                
        
        nonsepsis.append(features.to_numpy())
        
        if count==600:
          #  break
        count+=1
    else:
        print (filename)

nonsepsis = np.array(nonsepsis)    
print(nonsepsis.shape)
#array_has_nan = np.isnan(nonsepsis)
#print(array_has_nan)
print(nonsepsis[2])

"""nonsepsis[nonsepsis=0] = 'A'
nonsepsis[nonsepsis=1] = 'A+'
nonsepsis[nonsepsis=] = 'A'
nonsepsis[nonsepsis=0] = 'A'

"""

sepsis_dir='CinC2019 Dataset/training_setA/trainingA_sepsis_15h'

sepsis = []

count = 0
for filename in os.listdir(sepsis_dir):
    print (count)
    # read file as pandas dataframe
    df = pd.read_csv(sepsis_dir+'/'+filename,sep='|')
    #df_features = df[['HR','O2Sat','SBP','MAP','DBP','Resp','SepsisLabel']]
    df_features = df[['O2Sat','Resp','SepsisLabel']]
    i = df_features[df_features['SepsisLabel']==1]
    df_features = df_features.drop('SepsisLabel',axis = 1)
    
    sepsis_start_time_hr = i.index[0]
    monitoring_window_start = sepsis_start_time_hr - prediction_time_hr - time_window_hr
    monitoring_window_end = monitoring_window_start + time_window_hr
    
    df_features_12h = df_features.iloc[monitoring_window_start:monitoring_window_end,:]
    df_features_12h.fillna(method='ffill', inplace=True, limit=2) #replacing Nan with previous value
    df_features_12h.fillna(method='bfill', inplace=True, limit=2) #replacing Nan with next value (if first row is Nan)
    #print(sepsis_start_time_hr,filename, df_features_12h)
    
    if (df_features_12h.isnull().values.any() == True):
            continue
    
    if QUANTIZE == 1:
        #Quantizing according to NEWS2
        df_features_12h = quantize_news2(df_features_12h)
    
    # Feature construction
    features = pd.DataFrame(columns=df_features_12h.columns)
    features = features.append(df_features_12h.iloc[0])
    #print(features)

    if (df_features_12h.shape[0] > 1):
        i = 1
        
        while(i < df_features_12h.shape[0]): # find delt from first observation row and rate of change
            delta_from_first_obs = features.iloc[0] - df_features_12h.iloc[i]
            rate_delta_from_first_obs = delta_from_first_obs / i # rate of change
            
            features = features.append(delta_from_first_obs, ignore_index=True)
            features = features.append(rate_delta_from_first_obs, ignore_index=True)
            i+=1
        

        i=2
        while(i < df_features_12h.shape[0]):
            delta_consecutive = df_features_12h.iloc[i-1] - df_features_12h.iloc[i]
            features = features.append(delta_consecutive, ignore_index=True)
            i+=1
            
    features = features.append(df_features_12h.var(axis=0), ignore_index=True)
    
    sepsis.append(features.to_numpy())
    
    count+=1
    
sepsis = np.array(sepsis)    
print(sepsis.shape)

print(sepsis[1])
print(nonsepsis[1])


# create label array because all labels are zero
sepsis_label = np.zeros((nonsepsis.shape[0],1))
sepsis_label = np.append(sepsis_label, np.ones((sepsis.shape[0],1)), axis=0)
print(['label array:' , sepsis_label.shape])
print(sepsis_label.shape)
#combine datasets
combined_featureset = np.append(nonsepsis, sepsis, axis=0)
#print(combined_featureset[6])

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(combined_featureset,  np.ravel(sepsis_label), test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(np.count_nonzero(y_train))
print(np.count_nonzero(y_test))

#print(X_test[1])
#print(X_train[1])

n_train = X_train.shape[0]
input_shape = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(n_train, input_shape)

n_test = X_test.shape[0]
X_test = X_test.reshape(n_test, input_shape)

#y_train = np.reshape(y_train, (-1,1))
#y_test = np.reshape(y_test, (-1,1))
print(y_train.shape)

# xgboost for classification
from numpy import asarray
from numpy import mean
from numpy import std
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot


import sys
import math
 
import numpy as np
from sklearn.model_selection import GridSearchCV 
 
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
 
 
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
        #self.params.update({'objective': 'binary:logistic'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)

clf = XGBoostClassifier(
        eval_metric = 'auc',
        num_class = 2,
        nthread = 4,
        silent = 0,
        )

parameters = {
    'num_boost_round': [50, 100, 250],
    'eta': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9, 12],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.3, 0.5, 0.9, 1.0],
}
gridModel = GridSearchCV(clf, parameters, n_jobs=-1, cv=3, verbose=3)

gridModel.fit(X_train, y_train)

import pickle

# save the model to disk
model_filename = 'XGB-Models/' + 'XGB-Model-Temp-trainedonQuadset-Feb22-L' + str(prediction_time_hr) + '-M' + str(time_window_hr) +'-trainedagain_Feb22_for SHap'+ '.sav'
pickle.dump(gridModel, open(model_filename, 'wb'))

import pickle
# load the model from disk
loaded_model = pickle.load(open('XGB-Models/XGB-Model-Temp-trainedonQuadset-Feb22-L6-M4-trainedagain_Feb22_for SHap.sav', 'rb'))

# print best parameter after tuning 
print(loaded_model.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(loaded_model.best_estimator_)
 
# make predictions for test data
y_pred_proba = loaded_model.predict_proba(X_test)
y_pred = loaded_model.predict(X_test)

# print classification report 
print(classification_report(y_test, y_pred)) 

#confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
#print(y_pred_proba)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

## Plot ROC-AUC curves
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = loaded_model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='GBC')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
