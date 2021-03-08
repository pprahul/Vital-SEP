import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

VERBOSE = 0

time_window_hr = 4
prediction_time_hr = 6
nonsepsis_start_time_hr = 0

monitoring_window_start = nonsepsis_start_time_hr
monitoring_window_end = nonsepsis_start_time_hr + time_window_hr

nonsepsis_dir='trainingB_nonsepsis'

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
        df_features = df[['HR','O2Sat','Resp','Temp']]

        df_features_12h = df_features.iloc[monitoring_window_start:monitoring_window_end,:]
        #print(df_features_12h)
        df_features_12h.fillna(method='ffill', inplace=True, limit=2) #replacing Nan with previous value
        df_features_12h.fillna(method='bfill', inplace=True, limit=2) #replacing Nan with next value (if first row is Nan)
        #print(df_features_12h)
        if (df_features_12h.isnull().values.any() == True):
            continue
        
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
        
        if count==580:
            break
        count+=1
    else:
        continue

        
sepsis_dir='trainingB_sepsis_15h'

sepsis = []

count = 0
for filename in os.listdir(sepsis_dir):
    print (count)
    # read file as pandas dataframe
    df = pd.read_csv(sepsis_dir+'/'+filename,sep='|')
    #df_features = df[['HR','O2Sat','SBP','MAP','DBP','Resp','SepsisLabel']]
    df_features = df[['HR','O2Sat','Resp','Temp','SepsisLabel']]
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

# create label array because all labels are zero
sepsis_label = np.zeros((nonsepsis.shape[0],1))
sepsis_label = np.append(sepsis_label, np.ones((sepsis.shape[0],1)), axis=0)

#combine datasets
combined_featureset = np.append(nonsepsis, sepsis, axis=0)
sepsis_label = np.ravel(sepsis_label)

print(['label array:' , sepsis_label.shape])
print(['combined_featureset: ', combined_featureset.shape])

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X_test = combined_featureset
y_test = sepsis_label

n_test = X_test.shape[0]
input_shape = X_test.shape[1]*X_test.shape[2]
X_test = X_test.reshape(n_test, input_shape)

print(X_test.shape)

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

import pickle

model_filename = 'trained-models/XGBoost/XGB-Model-PPG-RR-Temp-L6-M4-verified.sav'

# load the model from disk
loaded_model = pickle.load(open(model_filename, 'rb'))


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
print(y_pred_proba)
print(y_pred)

#####from confusion matrix calculate accuracy
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity1)

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
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='black')
pyplot.plot(lr_fpr, lr_tpr, label='Tele-SEP', color='green')
pyplot.fill_between(lr_fpr,  lr_tpr-(0.07*lr_tpr), lr_tpr+(0.02*lr_tpr), alpha=0.3, color='green')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
