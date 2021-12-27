#!/usr/bin/env python
# coding: utf-8

# In[179]:


from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,balanced_accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve,matthews_corrcoef


# In[112]:


#데이터로드
pd.set_option("display.max_row", 100)
pd.set_option("display.max_column", 100)
import os
os.chdir(r'C:\Users\PC\Desktop')
data=pd.read_csv('hyper.csv')


# In[113]:


data.info()


# In[118]:


#문자열 수치형으로 전처리
def encode_features(dataDF):
    features = ['target','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery',
                'query_hypothyroid','query_hyperthyroid','pregnant','sick','tumor','lithium','goitre',
                'TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
        
    return dataDF

data = encode_features(data)
data.head()


# In[124]:


#나머지 변수들도 수치로 바꾸기
data=data.apply(pd.to_numeric, errors = 'coerce') 


# In[125]:


#null 값 최빈값으로 채우기
data[['target','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery',
                'query_hypothyroid','query_hyperthyroid','pregnant','sick','tumor','lithium','goitre',
                'TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']] = data[['target','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery',
                'query_hypothyroid','query_hyperthyroid','pregnant','sick','tumor','lithium','goitre',
                'TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']].fillna(data[['target','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery',
                'query_hypothyroid','query_hyperthyroid','pregnant','sick','tumor','lithium','goitre',
                'TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured']].mode()) 
#null값 평균으로 채우기
data[['age','TSH','T3','TT4','T4U','FTI']] = data[['age','TSH','T3','TT4','T4U','FTI']].fillna(data[['age','TSH','T3','TT4','T4U','FTI']].mean()) 


# In[126]:


#데이터 분할
y = data['target']
x= data.drop('target',axis=1)
x_train,x_test, y_train, y_test = train_test_split(x,y,random_state=42)


# In[129]:


#MinMax-범주형 아닌 수치형 데이터만 스케일링
scaler = MinMaxScaler()
scaler.fit(x_train[['age','TSH','T3','TT4','T4U','FTI']])
x_train[['age','TSH','T3','TT4','T4U','FTI']] = scaler.transform(x_train[['age','TSH','T3','TT4','T4U','FTI']])
scaler.fit(x_test[['age','TSH','T3','TT4','T4U','FTI']])
x_test[['age','TSH','T3','TT4','T4U','FTI']] = scaler.transform(x_test[['age','TSH','T3','TT4','T4U','FTI']])


# In[185]:


#평가지표 함수 생성
def get_eval(y_test,pred):
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test,pred)
    scores = roc_auc_score(y_test,pred)
    matthews=matthews_corrcoef(y_test,pred)
    balance=balanced_accuracy_score(y_test,pred)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},           F1: {3:.4f}, AUC:{4:.4f}, matt:{5:.4f}, balance:{6:.4f}'.format(accuracy, precision,recall,
                                                                          f1, scores,matthews, balance)) 
        


# In[131]:


#decision tree
dt_clf=DecisionTreeClassifier(random_state=42)
dt_clf.fit(x_train,y_train)
pred=dt_clf.predict(x_test)
print(accuracy_score(y_test, pred)) #단순히 예측하였을 때
#----------------------최적의 파라미터--------
params={'min_impurity_decrease': np.arange(0.0001,0.001,0.0001),'max_depth': range(5,20,1),'min_samples_split': range(2,100,10)}
gs=GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1,cv=10)
gs.fit(x_train,y_train)
#min_impurity_decrease 바꿔가며 총 5번 실행, cv=10, 총25개의 모델 훈련 


# In[187]:


dt=gs.best_estimator_
print(dt)
print(dt.score(x_train,y_train))
#test 셋에 모델 테스트하기
pred = dt.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# ##10번의 교차검증을 하고 최적의 파라미터를 확인하여 decision tree 모델을 돌렸을 때 정확도가 더 높은 것을 알 수 있다.

# In[193]:


#로지스틱회귀-패널티 none일 때
lr_clf= LogisticRegression(penalty='none')
lr_clf.fit(x_train, y_train)
lr_preds= lr_clf.predict(x_test)
print(accuracy_score(y_test, lr_preds)) #단순히 예측 했을 때 
#----------------------최적의 파라미터--------
from sklearn.model_selection import GridSearchCV
params={'C': [0.01,0.1,1,1,5,10]}
grid_clf=GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=10)
grid_clf.fit(x_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.best_score_) #10번 교차검증했을 때의 정확도
ypred=grid_clf.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# In[195]:


#로지스틱회귀-패널티 릿지일 때
lr_clf= LogisticRegression(penalty='l1',solver='saga')
lr_clf.fit(x_train, y_train)
lr_preds= lr_clf.predict(x_test)
print(accuracy_score(y_test, lr_preds)) #단순히 예측 했을 때 
#----------------------최적의 파라미터--------
from sklearn.model_selection import GridSearchCV
params={'C': [0.01,0.1,1,1,5,10]}
grid_clf=GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=10)
grid_clf.fit(x_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.best_score_) #10번 교차검증했을 때의 정확도
ypred=grid_clf.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# In[189]:


#로지스틱회귀-패널티 라쏘일 때
lr_clf= LogisticRegression(penalty='l2')
lr_clf.fit(x_train, y_train)
lr_preds= lr_clf.predict(x_test)
print(accuracy_score(y_test, lr_preds)) #단순히 예측 했을 때 
#----------------------최적의 파라미터--------
from sklearn.model_selection import GridSearchCV
params={'C': [0.01,0.1,1,1,5,10]}
grid_clf=GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=10)
grid_clf.fit(x_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.best_score_) #10번 교차검증했을 때의 정확도
ypred=grid_clf.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# In[190]:


#로지스틱회귀-패널티 일레스틱넷일 때
#단순히 모델을 돌렸을때
lr_clf= LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5)
lr_clf.fit(x_train, y_train)
lr_preds= lr_clf.predict(x_test)
print(accuracy_score(y_test, lr_preds)) 
#----------------------최적의 파라미터--------
from sklearn.model_selection import GridSearchCV
params={'C': [0.01,0.1,1,1,5,10]}
grid_clf=GridSearchCV(lr_clf, param_grid=params, scoring='accuracy', cv=10)
grid_clf.fit(x_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.best_score_) #10번 교차검증했을 때의 정확도
ypred=grid_clf.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# ##최적의 파라미터를 찾아 학습 데이터에 학습시킨 후의 정확도가 더 높은 것을 알 수 있다.

# In[196]:


#배깅-랜덤포레스트
params={'n_estimators':[3,10,50,100],'max_depth': [6,8,10,12],'min_samples_split': [8,16,20],'min_samples_leaf': [8,12,18]}
rf_clf=RandomForestClassifier(random_state=20, n_jobs=-1)
grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=10, n_jobs=-1)
grid_cv.fit(x_train,y_train)
print(grid_cv.best_params_,grid_cv.best_score_)
#별도의 테스트데이터 세트에서 예측성능 측정
rf_clf2=RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=8, min_samples_split=20, random_state=20)
rf_clf2.fit(x_train,y_train)
pred=rf_clf2.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# In[191]:


#ERT
params={'n_estimators':[3,10,100,300],'max_depth': [6,8,10,12],'min_samples_split': [8,16,20],'min_samples_leaf': [8,12,18]}
ert_clf=ExtraTreesClassifier(random_state=20, n_jobs=-1)
grid_cv=GridSearchCV(ert_clf, param_grid=params, cv=10, n_jobs=-1)
grid_cv.fit(x_train,y_train)
print(grid_cv.best_params_,grid_cv.best_score_)
#별도의 테스트데이터 세트에서 예측성능 측정
rf_clf2=RandomForestClassifier(n_estimators=10, max_depth=8, min_samples_leaf=8, min_samples_split=8, random_state=20)
rf_clf2.fit(x_train,y_train)
pred=rf_clf2.predict(x_test)
get_eval(y_test,pred) #예측한 결과값


# ##정확도와 정밀도,f1,AUC,matthews, balance score은 decision tree 모델과 로지스틱에 라쏘, 일레스틱으로 패널티 준 모델이 각각 0.9848, 09881, 0.9920, 0.8882, 0.8371, 0.8882로 제일 높았다. 
# 
# ##재현율로는 로지스틱회귀에 릿지, 라쏘로 패널티를 준 모형, 랜덤포레스트, ERT가 0.9987로 가장 높았다.
