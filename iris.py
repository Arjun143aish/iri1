import os
import numpy as np
import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(iris,test_size = 0.3, random_state =123)

Train_X  =Train.drop(['species'], axis =1)
Train_Y = Train['species'].copy()
Test_X = Test.drop(['species'], axis =1)
Test_Y  =Test['species'].copy()

from sklearn.linear_model import LogisticRegression

M1 = LogisticRegression(random_state=123).fit(Train_X,Train_Y)

Test_pred = M1.predict(Test_X)
Test_pred

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_pred,Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

print(M1.predict([[5.8,2.8,5.1,2.4]]))

import pickle

pickle.dump(M1, open('model1.pkl','wb'))
model1 = pickle.load(open('model.pkl','rb'))

print(model1.predict([[5.8,2.8,5.1,2.4]]))


