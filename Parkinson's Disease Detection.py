#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
import seaborn as sns


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import os


# In[65]:


df=pd.read_csv(r'C:/Users/USER/Downloads/parkinsons.data')
print('Dataset is being read')


# In[66]:


df.head()


# In[67]:


df.tail()


# In[30]:


# Check if any of the columns have null values
print(df.isnull().sum())


# In[68]:


features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
scaler=MinMaxScaler((-1,1))
print('The sample data set :')
print(df.head)
print(df.dtypes)


# In[69]:


print('------------DESCRIPTIVE STATISTICS---------------')
print(df.describe())

import seaborn as sb
import matplotlib.pyplot as plt


# In[70]:


print(f"The shape of the DatFrame is: {df.shape}, which means there are {df.shape[0]} rows and {df.shape[1]} columns.")


# In[34]:


df_summary = df.describe()
df_summary


# In[35]:


corr_map=df.corr()
plt.title('Correaltion heatmap')
sb.heatmap(corr_map,square=True)
plt.show()


# In[36]:


#A function that returns value counts for a column split by status (univariate analysis)
def groupby_get_cc_count(tdf, col):
    tdf = tdf.groupby([col, "status"])["status"].count().reset_index(level = 0)
    tdf.columns = [col, "count"]
    tdf = tdf.reset_index()
    return tdf


# In[37]:


df[["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "status"]]


# In[38]:


print('----------------DATASET CLEANING-------------------')
print('Checking for null values:')
print(df.isna().sum())


# In[39]:


#univariate and bivariate
# A function that returns value counts for a column split by status
def groupby_get_cc_count(tdf, col):
    tdf = tdf.groupby([col, "status"])["status"].count().reset_index(level = 0)
    tdf.columns = [col, "count"]
    tdf = tdf.reset_index()
    return tdf


# In[40]:


df[["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "status"]]


# In[41]:


def draw_axvlines(plt, col):
    mean = df_summary.loc["mean", col]
    q1 = df_summary.loc["25%", col]
    q2 = df_summary.loc["50%", col]
    q3 = df_summary.loc["75%", col]
    plt.axvline(mean, color = "g");              # Plotting a line to mark the mean 
    plt.axvline(q1, color = "b");                # Plotting a line to mark Q1 
    plt.axvline(q2, color = "navy");             # Plotting a line to mark Q2 
    plt.axvline(q3, color = "purple");           # Plotting a line to mark Q3
    plt.legend({"Mean": mean, "25%" : q1, "50%" : q2, "75%" : q3});

fig, axes = plt.subplots(3, 2, figsize = (20,15));
fig.suptitle('Distribution charts for Age, Experience and income.');


# Create boxplot to show distribution of Age
sns.boxplot(df["MDVP:Fo(Hz)"], ax = axes[0][0], color = "mediumslateblue");
axes[0][0].set(xlabel = 'Distribution of Age');

pp = sns.distplot(df["MDVP:Fo(Hz)"], ax = axes[0][1], bins = 10, color = "mediumslateblue");
axes[0][1].set(xlabel = 'Distribution of Age');
draw_axvlines(pp, "MDVP:Fo(Hz)");


# Create boxplot to show distribution of creatinine_phosphokinase
sns.boxplot(df["MDVP:Fhi(Hz)"], ax = axes[1][0], color = "mediumslateblue");
axes[1][0].set(xlabel = 'Distribution of creatinine_phosphokinase');

pp = sns.distplot(df["MDVP:Fhi(Hz)"], ax = axes[1][1], bins = 10, color = "mediumslateblue");
axes[1][1].set(xlabel = 'Distribution of creatinine_phosphokinase');
draw_axvlines(pp, "MDVP:Fhi(Hz)")


# Create boxplot to show distribution of platelets
sns.boxplot(df["MDVP:Flo(Hz)"], ax = axes[2][0], color = "mediumslateblue");
axes[2][0].set(xlabel = 'Distribution of platelets');

pp = sns.distplot(df["MDVP:Flo(Hz)"], ax = axes[2][1], color = "mediumslateblue");
axes[2][1].set(xlabel = 'Distribution of platelets');
draw_axvlines(pp, "MDVP:Flo(Hz)")


# In[42]:


plt.figure(figsize = (15, 5))
sns.scatterplot(x = "MDVP:Fo(Hz)", y = "MDVP:Jitter(%)", data = df[["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "status"]], hue = "status", alpha = 0.5);


# In[71]:


x=scaler.fit_transform(features)
y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[72]:


n_neighbors=5
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[73]:


models = []
models.append(('LogisticRegression',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier(n_neighbors=n_neighbors)))
models.append(('SVC',SVC()))
models.append(('DecisionTree',DecisionTreeClassifier()))
models.append(('NaiveBayes',GaussianNB()))


# In[74]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
error='accuracy'


# In[75]:


for name,model in models:
    fold=KFold(n_splits=10,random_state=None)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f "%(name,result.mean())
    print(msg)


# In[48]:


fig=plt.figure()
fig.suptitle("Comparing algorithms")
plt.boxplot(predictions)
plt.xlabel(names)
plt.show()


# In[76]:


x=scaler.fit_transform(features)
y=labels
print('Feature scaling is being performed...')


# In[50]:


#KNN ALGORITHM
print('------------KNN CLASSIFICATION ALGORITHM---------------')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
variance = pca.explained_variance_ratio_
classi = KNeighborsClassifier(n_neighbors = 8 , p=2, metric ='minkowski')
classi.fit(x_train,y_train)
y_pred = classi.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('Confusion  matrix of KNN algorithm:')
print(cm)
asc=accuracy_score(y_test,y_pred)
print('The accuracy score of the KNN classification algorithm is %.2f'%asc)


# In[77]:


#RANDOM FOREST ALGORITHM
print('------------RANDOM FOREST CLASSIFIER---------------')
x=df.drop('status',axis=1)
x=x.drop('name',axis=1)
y=df['status']

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=1)
random_forest.fit(x_train,y_train)
y_predict = random_forest.predict(x_test)

cmm=confusion_matrix(y_test,y_predict)

print('Confusion  matrix of Random Forest Classifier:')
print(cmm)
acs = accuracy_score(y_test,y_predict)
print('The accuracy score of the Random Forest Classifier is %.2f'%acs)


# In[78]:


#XGB
print('------------XTREME GRADIENT BOOST ALGORITHM---------------')
from xgboost import XGBClassifier
import warnings
model1 = XGBClassifier()
x=scaler.fit_transform(features)
y=labels
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
warnings.filterwarnings('ignore')
modelx=XGBClassifier(eval_metric='mlogloss')
modelx.fit(x_train,y_train)
y_predx=modelx.predict(x_test)


# In[79]:


#Output = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',gamma=0, gpu_id=-1, importance_type='gain',interaction_constraints='', learning_rate=0.300000012,max_delta_step=0, max_depth=6, min_child_weight=1, missing=np.nan,monotone_constraints='()', n_estimators=100, n_jobs=4,num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,scale_pos_weight=1, subsample=1, tree_method='exact',use_label_encoder=False,validate_parameters=1, verbosity=None)

cmma=confusion_matrix(y_test,y_predx)
print('Confusion  matrix of XGBoost algorithm is:')
print(cmma)
acsc = accuracy_score(y_test,y_predx)
print('The accuracy score of the XGBoost Classifier is %.2f'%acsc)


# In[80]:


#SVM
print('------------SUPPORT VECTOR MACHINES---------------')
from sklearn.svm import SVC
classifi2 = SVC()

classifi2.fit(x_train,y_train)
y2_pred = classifi2.predict(x_test)
cma=confusion_matrix(y_test,y2_pred)
print('Confusion  matrix of Support vector machine:')
print(cma)

acc = accuracy_score(y_test,y2_pred)
print('The accuracy score of the SVM is %.2f'%acc)
	
print('--------------------------------------------------')
print('Of all the classification algorithms fitted ,XGBoost classifier had the maximum accuracy score')


# In[1]:


# SVM_PREDICTIVE_ANALYSIS


# In[81]:


X = df.drop(columns=['name','status'], axis=1)
Y = df['status']


# In[82]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[83]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[84]:


model = svm.SVC(kernel='linear')


# In[85]:


# training the SVM model with training data
model.fit(X_train, Y_train)


# In[86]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")


# In[87]:


input_data = (95.730,132.068,91.754,0.00551,0.00006,0.00293,0.00332,0.00880,0.02093,0.191,0.01073,0.01277,0.01717,0.03218,0.01070,21.812,0.615551,0.773587,-5.498678,0.327769,2.322511,0.231571)

# changing input data to numpy array
input_data_numpy = np.asarray(input_data)

#reshaping the numpy array 
input_data_reshape = input_data_numpy.reshape(1,-1)

#standardizing the input data 
std_data = scaler.transform(input_data_reshape)

## prediction
prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 1):
  print('The patient has Parkinson')
elif (prediction[0] == 0):
  print('The patient does not have Parkinson')
else:
  print('Some error in processing')


# In[ ]:




