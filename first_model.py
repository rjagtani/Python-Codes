#### first_python_script

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
os.chdir('C:\\Users\\rjagtani\\Desktop\\Python')
titanic=pd.read_csv('train_titanic.csv')
lin=LogisticRegression()
Y=titanic.loc[:,'Survived']
X=titanic.loc[:,'Fare']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
lin.fit(x_train,y_train)
y_pred=lin.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lin.score(x_test, y_test)))
lin.score(x_test,y_test)
print(classification_report(y_test,y_pred))