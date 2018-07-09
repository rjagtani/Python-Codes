#os.chdir('C:\\Users\\rjagtani')
import keras
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.datasets import fashion_mnist
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit



### Changing 

os.chdir('C:\\Users\\rjagtani\\Desktop\\Python')
bcancer=pd.read_csv('bcancer.csv')
bcancer_x=np.array(bcancer.drop('y_flag',axis=1))
bcancer_y=np.array(bcancer.loc[:,'y_flag'])
train_x,test_x,train_y,test_y=train_test_split(bcancer_x,bcancer_y,test_size=0.3,random_state=42)

######### Trying Logistic Regression

clf=LogisticRegression()
clf.fit(train_x,train_y)
predict_test=clf.predict(test_x)
confusion_matrix(test_y,predict_test)
print(classification_report(test_y,predict_test))
prob=pd.DataFrame(clf.predict_proba(test_x))
prob.describe()


######### Trying Random Forest

rf=RandomForestClassifier(n_estimators=40,max_features=4)
rf.fit(train_x,train_y)
predict_test=rf.predict(test_x)
confusion_matrix(test_y,predict_test)
print(classification_report(test_y,predict_test))
prob=pd.DataFrame(rf.predict_proba(test_x))
prob.describe()


########### Trying Gradient Boosting

gbm=GradientBoostingClassifier()
gbm.fit(train_x,train_y)
predict_test=gbm.predict(test_x)
confusion_matrix(test_y,predict_test)
print(classification_report(test_y,predict_test))
prob=pd.DataFrame(gbm.predict_proba(test_x))
prob.describe()


############### Trying Neural Net

bcancer_y1=to_categorical(bcancer_y,2)
train_x,test_x,train_y,test_y=train_test_split(bcancer_x,bcancer_y,test_size=0.3,random_state=42)
ann=Sequential()
ann.add(Dense(3,activation='relu',input_shape=(30,)))
ann.add(Dense(1,activation='sigmoid'))
ann.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(train_x,train_y,epochs=20)

################# GridSearchCV & GBM

gbm=GradientBoostingClassifier()
parameters={'learning_rate':[0.01,0.05,0.1],'n_estimators':[50,100,150]}
grid=GridSearchCV(gbm,param_grid=parameters,cv=10,scoring=['accuracy'],refit='accuracy',n_jobs=2)
grid.fit(train_x,train_y)
best_gbm=grid.best_estimator_
predict_test=best_gbm.predict(test_x)
print(classification_report(test_y,predict_test))
confusion_matrix(test_y,predict_test)
accuracy_score(test_y,predict_test)
grid_results=pd.DataFrame(grid.cv_results_)
grid_results1=pd.DataFrame(grid.cv_results_)

######################### RandomSearchCV & GBM

from scipy.stats import randint as sp_randint
gbm=GradientBoostingClassifier()
parameters={'learning_rate':[0.0001,1],'n_estimators':sp_randint(100,1000)}
grid=RandomizedSearchCV(gbm,parameters,n_iter=30)
grid.fit(train_x,train_y)
grid.best_params_
best_gbm=grid.best_estimator_
predict_test=best_gbm.predict(test_x)
print(classification_report(test_y,predict_test))
confusion_matrix(test_y,predict_test)
accuracy_score(test_y,predict_test)

################################

import dfply 
from dfply import *
bcancer.columns=[k.replace(" ","_") for k in bcancer.columns]
bc1 = (bcancer >> select(X.mean_radius,X.mean_texture)) >> mask(X.mean_radius<13) >> mutate(new_col=X.mean_radius + X.mean_texture)
bc1.flag=pd.Series([1 if k>11 else 0 for k in bc1.mean_radius])

def my_func(row):
    return np.average(row)
    
print(bc1.apply(lambda x: 1 if x['mean_radius'] > 11 else 0,axis=1))
#bc1.apply()

@make_symbolic
def my_func(x,t):
    return [1 if k>t else 0 for k in x]

bc1 = (bcancer >> select(X.mean_radius,X.mean_texture)) >> mask(X.mean_radius<13) >> mutate(new_col=X.mean_radius + X.mean_texture) >> mutate(flag=my_func(X.mean_radius,12))
bc1 = bc1.reset_index().drop('index',1)
bc1.loc[0:63,'flag2']=1
bc1.loc[64:127,'flag2']=2
bc1.loc[128:255,'flag2']=3
bc1.flag2.value_counts()

bc1.loc[:,'gg']=1

bc1.iloc[0:63,4]=1
bc1.iloc[64:127,4]=2
bc1.iloc[128:255,4]=3
bc1.gg.value_counts()

bc2 = (bc1 >> group_by(X.flag) >> mean()) >> 