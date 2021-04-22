import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_recall_curve, make_scorer, \
    plot_confusion_matrix
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
raw_data = pd.read_csv('application_train.csv')
# print(raw_data.isnull().values.any())
raw_data.replace("", "NaN", inplace=True)
raw_data.dropna(axis='columns', thresh=300000, inplace=True)
# print(raw_data.isnull().values.any())
# print(raw_data)
raw_data.fillna(method='backfill', inplace=True)
# print(raw_data.isnull().values.any())
# convert strings to numerical variables
raw_data.loc[raw_data.NAME_CONTRACT_TYPE == 'Cash loans', 'NAME_CONTRACT_TYPE'] = 1
raw_data.loc[raw_data.NAME_CONTRACT_TYPE == 'Revolving loans', 'NAME_CONTRACT_TYPE'] = 0
raw_data.loc[raw_data.CODE_GENDER == 'M', 'CODE_GENDER'] = 1
raw_data.loc[raw_data.CODE_GENDER == 'F', 'CODE_GENDER'] = 0
raw_data.loc[raw_data.CODE_GENDER == 'XNA', 'CODE_GENDER'] = 0
raw_data.loc[raw_data.FLAG_OWN_CAR == 'Y', 'FLAG_OWN_CAR'] = 1
raw_data.loc[raw_data.FLAG_OWN_CAR == 'N', 'FLAG_OWN_CAR'] = 0
raw_data.loc[raw_data.FLAG_OWN_REALTY == 'Y', 'FLAG_OWN_REALTY'] = 1
raw_data.loc[raw_data.FLAG_OWN_REALTY == 'N', 'FLAG_OWN_REALTY'] = 0

# print(raw_data.head())
# variable selection frist
temp_data = pd.DataFrame(raw_data,
                         columns=['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', \
                                  'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', \
                                  'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', \
                                  'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS'])
temp_data.astype(float)
print(temp_data.dtypes)

# correlation heatmap

matrix = np.triu(temp_data.corr())
plt.subplots(figsize=(25, 20))
sns.heatmap(temp_data.corr(), annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=matrix)
plt.show()

final_feature = pd.DataFrame(temp_data,
                             columns=['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', \
                                      'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'REGION_POPULATION_RELATIVE', 'DAYS_EMPLOYED', \
                                      'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS'])

# final_feature.astype(float)
print(final_feature.dtypes)

x = final_feature.drop('TARGET', axis=1)
y = final_feature['TARGET']
# x.astype(float)
print(x.dtypes)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

dtree = DecisionTreeClassifier()
dtree_param = {'criterion': ['gini', 'entropy'],
               'splitter': ['random'],
               'max_depth': [25, None],
               'random_state': [12345],
               'max_features': ['log2'],
               'max_leaf_nodes': [20]}

train_data = x_train.copy()
train_data['TARGET'] = y_train
bad_data = train_data[train_data['TARGET'] == 0]
bad_size = len(bad_data)

minor_size = int(bad_size)
non_bad_data = train_data[train_data['TARGET'] == 1].sample(n=minor_size, random_state=1234, replace=True)
SMOTE_data = pd.concat([bad_data, non_bad_data])
SMOTE_x = SMOTE_data.drop("TARGET", axis=1)
SMOTE_y = SMOTE_data['TARGET']
sns.countplot(SMOTE_y)
plt.show()

SMOTE_x_train = SMOTE_x
SMOTE_y_train = SMOTE_y
SMOTE_x_test = x_test
SMOTE_y_test = y_test

recall_score = make_scorer(recall_score)
SMOTE_dtree = GridSearchCV(dtree, param_grid=dtree_param, scoring=recall_score)
SMOTE_dtree.fit(SMOTE_x_train, SMOTE_y_train)

SMOTE_dtree_prediction = SMOTE_dtree.predict(SMOTE_x_test)
SMOTE_dtree_prob = SMOTE_dtree.predict_proba(SMOTE_x_test)[:, 1]
print(SMOTE_dtree_prob)
print(classification_report(SMOTE_y_test, SMOTE_dtree_prediction))

from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, plot_roc_curve

ax = plt.gca()
plot_roc_curve(SMOTE_dtree, SMOTE_x_test, SMOTE_y_test, ax=ax, name='Decision Tree')
plt.show()

import pickle

s = pickle.dumps(SMOTE_dtree)
with open('myModel.model', 'wb+') as f:
    f.write(s)
