
# coding: utf-8

# In[1]:


# Load required libraries and data:

import numpy as np
import pandas as pd


train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)

test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'wage_class']

train_set.columns = col_labels
test_set.columns= col_labels


# In[2]:


train_set.shape, test_set.shape


# In[3]:


train_set.sample(4, random_state = 42)


# In[4]:


test_set.sample(4, random_state= 42)


# In[5]:


train_set.isnull().sum()


# In[6]:


test_set.isnull().sum()


# In[7]:


pd.DataFrame([train_set.dtypes, test_set.dtypes], index = ['train_set', 'test_set']).T


# In[8]:


# Find the columns having data types as object:


# In[9]:


for i in train_set.columns:
    if train_set[i].dtypes == 'object':
        print(i)


# In[10]:


train_set.workclass.value_counts()


# In[11]:


train_set


# In[12]:


train_set.relationship.value_counts()


# In[13]:


train_set.workclass.nunique(),train_set.education.nunique(),train_set.marital_status.nunique(),train_set.native_country.nunique()


# In[14]:


X_train = train_set.copy()
X_test = test_set.copy()


# In[15]:


X_train.columns


# In[16]:


# Converting Categorical Values to Numeric Values:


# In[17]:


dict_sex = {}
count = 0
for i in X_train.sex.unique():
    dict_sex[i] = count
    count +=1


# In[18]:


dict_workclass ={}
count = 0
for i in X_train.workclass.unique():
    dict_workclass[i] = count
    count +=1


# In[19]:


dict_education = {}
count = 0
for i in X_train.education.unique():
    dict_education[i] = count
    count +=1


# In[20]:


dict_marital_status = {}
count = 0
for i in X_train.marital_status.unique():
    dict_marital_status[i] = count
    count +=1


# In[21]:


dict_occupation = {}
count = 0
for i in X_train.occupation.unique():
    dict_occupation[i] = count
    count +=1


# In[22]:


dict_relationship = {}
count = 0
for i in X_train.relationship.unique():
    dict_relationship[i] = count
    count +=1


# In[23]:


dict_race = {}
count = 0
for i in X_train.race.unique():
    dict_race[i] = count
    count +=1


# In[24]:


dict_native_country ={}
count = 0
for i in X_train.native_country.unique():
    dict_native_country[i] = count
    count +=1


# In[25]:


dict_wage_class = {}
count = 0
for i in X_train.wage_class.unique():
    dict_wage_class[i] = count
    count +=1


# In[26]:


dict_sex,dict_education,dict_wage_class,dict_native_country,dict_race,dict_occupation ,dict_marital_status 


# In[27]:


X_train['sex'] = X_train['sex'].map(dict_sex)
X_train['education'] = X_train['education'].map(dict_education)
X_train['wage_class'] = X_train['wage_class'].map(dict_wage_class)
X_train['native_country'] = X_train['native_country'].map(dict_native_country)
X_train['race'] = X_train['race'].map(dict_race)
X_train['occupation']=X_train['occupation'].map(dict_occupation)
X_train['marital_status'] = X_train['marital_status'].map(dict_marital_status)
X_train['workclass'] = X_train['workclass'].map(dict_workclass)
X_train['relationship'] = X_train['relationship'].map(dict_relationship)


# In[28]:


X_train.isnull().sum()


# In[29]:


Xtrain = X_train.astype(int)


# In[30]:


X_train.head()


# In[31]:


X_train.describe()


# In[32]:


dict_wage_class = {}
count = 0
for i in X_test.wage_class.unique():
    dict_wage_class[i] = count
    count +=1
    
dict_native_country ={}
count = 0
for i in X_test.native_country.unique():
    dict_native_country[i] = count
    count +=1


# In[33]:


X_test['sex'] = X_test['sex'].map(dict_sex)
X_test['education'] = X_test['education'].map(dict_education)
X_test['wage_class'] = X_test['wage_class'].map(dict_wage_class)
X_test['native_country'] = X_test['native_country'].map(dict_native_country)
X_test['race'] = X_test['race'].map(dict_race)
X_test['occupation']=X_test['occupation'].map(dict_occupation)
X_test['marital_status'] = X_test['marital_status'].map(dict_marital_status)
X_test['workclass'] = X_test['workclass'].map(dict_workclass)
X_test['relationship'] = X_test['relationship'].map(dict_relationship)


# In[34]:


dict_wage_class


# In[35]:


X_test.head()


# In[36]:


X_test.describe()


# In[37]:


# Annual Income Data Analysis using Visualization:


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(40, 20))
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.countplot('sex', data=train_set, hue='wage_class')


# In[39]:


g = sns.countplot('workclass', data=train_set, hue='wage_class')
g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 12)
plt.show()


# In[40]:


g = sns.countplot('education', data=train_set, hue='wage_class')
g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 12)
plt.show()


# In[41]:


g = sns.countplot('wage_class', data=train_set)
g.set_xticklabels(g.get_xticklabels(), rotation = 45, fontsize = 12)
plt.show()


# In[42]:


pd.DataFrame.hist(train_set, figsize = [15, 15])
plt.show()


# In[43]:


x_train = X_train.drop('wage_class',axis=1)
y_train = X_train['wage_class']

x_test = X_test.drop('wage_class',axis=1)
y_test = X_test['wage_class']


# In[44]:


X = x_train.values
Y = y_train.values
Xtest = x_test.values
Ytest = y_test.values


# In[45]:


x_train.shape, y_train.shape, X.shape, Y.shape, Xtest.shape, Ytest.shape


# In[46]:


Xtest


# In[47]:


# Using Boosting Method of Ensemble model to predict the annual income


# In[48]:


from xgboost.sklearn import XGBClassifier
#set the parameters for the xgbosst model
params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': 1.0,
    'n_estimators': 5
}
params['eval_metric'] = ['logloss', 'auc']


# In[49]:


# Train the XGBClassifier model:


# In[50]:


bst = XGBClassifier(**params).fit(X,Y)


# In[51]:


# Predict the annual income:


# In[52]:


preds = bst.predict(Xtest)
preds


# In[53]:


preds_proba = bst.predict_proba(Xtest)
preds_proba


# In[54]:


#Measure the accuracy of the model:


# In[55]:


correct = 0
from sklearn.metrics import accuracy_score
for i in range(len(preds)):
    if (y_test[i] == preds[i]):
        correct +=1
        
acc = accuracy_score(Ytest, preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Accuracy Score :{:.4f}'.format(acc))
print('Error: {0:.4f}'.format(1-acc))        


# In[56]:


from sklearn.metrics import classification_report
print(classification_report(Ytest, preds))


# In[57]:


# Confusion Matrix:


# In[58]:


import scikitplot
scikitplot.metrics.plot_confusion_matrix(Ytest, preds)


# In[60]:


# ROC:


# In[62]:


scikitplot.metrics.plot_roc_curve(Ytest, preds_proba)


# In[63]:


# Comapring the ensemble model with single random LogisticRegression Model:


# In[64]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,Y)


# In[65]:


pred_lr = lr.predict(Xtest)
pred_lr


# In[66]:


pred_lr_proba = lr.predict_proba(Xtest)
pred_lr_proba


# In[67]:


print('Accuracy score:{:.4f}'.format(accuracy_score(Ytest, pred_lr)))


# In[68]:


print(classification_report(Ytest, pred_lr))


# In[69]:


# Confusion Matrix:


# In[70]:


import scikitplot
scikitplot.metrics.plot_confusion_matrix(Ytest, pred_lr)


# In[71]:


# ROC:


# In[72]:


scikitplot.metrics.plot_roc_curve(Ytest,pred_lr_proba)


# In[73]:


# Using Bagging Method of Ensemble model and base model as Logistic Regression:


# In[74]:


from sklearn.ensemble import BaggingClassifier
bag_LR = BaggingClassifier(LogisticRegression(),
                            n_estimators=10, max_samples=0.5,
                            bootstrap=True, random_state=3)


# In[75]:


bag_LR.fit(X,Y)


# In[76]:


# Predictions by the Bagging Ensemble model:


# In[77]:


bag_preds = bag_LR.predict(Xtest)
bag_preds


# In[78]:


bag_preds_proba = bag_LR.predict_proba(Xtest)
bag_preds_proba


# In[79]:


# Score of the bagging ensemble model:


# In[80]:


bag_LR.score(Xtest, Ytest)


# In[81]:


print(accuracy_score(Ytest, bag_preds))


# In[82]:


# Confusion Matrix:


# In[83]:


scikitplot.metrics.plot_confusion_matrix(Ytest, bag_preds)


# In[84]:


# ROC:


# In[85]:


scikitplot.metrics.plot_roc_curve(Ytest, bag_preds_proba)


# In[ ]:


# CONCLUSION

#The boosting Classifier method yields a better performance in this scenario as compared to single random Binary classifier and 
# bagging method.

