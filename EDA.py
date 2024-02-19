#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 05:31:59 2023

@author: arnaudcruchaudet
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_path = '/Users/arnaudCruchaudet/Desktop/Combined_Data.xlsx'
df = pd.read_excel(file_path)
df = df.drop("Test", axis=1)

df.head()
df.nunique()
df.describe().T



# Case where nothing in toilet

df.query("`Urine level` == 0 & `Paper level` == 0 & `Feces Level` == 0")
df["Case of flush"].value_counts()



# Chi2 tests

def test_chi2(variable1, variable2, alpha=0.02):
    chi2, pvalue, dof, expected = stats.chi2_contingency(pd.crosstab(variable1, variable2))
    print(f"Null Hypothesis: 'There is no link between {variable1.name} and {variable2.name}'")
    if pvalue < alpha:
        print("Hypothesis rejected!")
    else:
        print("Hypothesis cannot be rejected")

variable1 = df["Feces Level"]
variable2 = df["Case of flush"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Urine level"]
variable2 = df["Case of flush"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Paper level"]
variable2 = df["Case of flush"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Paper level"]
variable2 = df["Urine level"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Feces Level"]
variable2 = df["Urine level"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Feces Level"]
variable2 = df["Paper level"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Flush volume"]
variable2 = df["Case of flush"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["number from 0 to 51"]
variable2 = df["Case of flush"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()

variable1 = df["Flush volume"]
variable2 = df["PCB value"]
test_chi2(variable1, variable2, alpha=0.02)
sns.heatmap(pd.crosstab(variable1, variable2), annot=True)
plt.show()



# Pearson correlation between photodiode

features = ["Blue LED 1\nPhotodiode 1", "Blue LED 1\nPhotodiode 2", 
            "Green LED 1\nPhotodiode 1", "Green LED 1\nPhotodiode 2", 
            "Red LED 1\nPhotodiode 1", "Red LED 1\nPhotodiode 2", 
            "Blue LED 2\nPhotodiode 1", "Blue LED 2\nPhotodiode 2", 
            "Green LED 2\nPhotodiode 1", "Green LED 2\nPhotodiode 2", 
            "Red LED 2\nPhotodiode 1", "Red LED 2\nPhotodiode 2"]


X = df[features]
y = df['Case of flush']

sns.heatmap(X.corr(), annot=True)



# VIF to reduce multicollinearity
model = sm.OLS(y, X)
results = model.fit()
vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

Z = df[features + ['Case of flush']]
A = Z.corr()


# VIF 1
X1 = df["Red LED 2\nPhotodiode 1"]
y1 = df["Green LED 2\nPhotodiode 1"]
model = sm.OLS(y, X)
results = model.fit()
res_GL2_1 = results.resid
df.loc[:,"Green LED 2\nPhotodiode 1"] = res_GL2_1

features = ["Blue LED 1\nPhotodiode 1", "Blue LED 1\nPhotodiode 2", 
            "Green LED 1\nPhotodiode 1", "Green LED 1\nPhotodiode 2", 
            "Red LED 1\nPhotodiode 1", "Red LED 1\nPhotodiode 2", 
            "Blue LED 2\nPhotodiode 1", "Blue LED 2\nPhotodiode 2", 
            "res_GL2_1", "Green LED 2\nPhotodiode 2", 
            "Red LED 2\nPhotodiode 1", "Red LED 2\nPhotodiode 2"]
X = df[features]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
Z1 = df[features + ['Case of flush']]
A1 = Z1.corr()


# VIF 2
X2 = df["Red LED 1\nPhotodiode 1"]
y2 = df["Green LED 1\nPhotodiode 1"]
model = sm.OLS(y1, X1)
results = model.fit()
res_GL1_1 = results.resid
df["res_GL1_1"] = res_GL1_1  # Ajoutez cette ligne pour créer la nouvelle colonne

features = ["Blue LED 1\nPhotodiode 1", "Blue LED 1\nPhotodiode 2",
           "res_GL1_1", "Green LED 1\nPhotodiode 2", 
           "Red LED 1\nPhotodiode 1", "Red LED 1\nPhotodiode 2", 
           "Blue LED 2\nPhotodiode 1", "Blue LED 2\nPhotodiode 2", 
           "res_GL2_1", "Green LED 2\nPhotodiode 2", 
           "Red LED 2\nPhotodiode 1", "Red LED 2\nPhotodiode 2"]
X = df[features]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

Z2 = df[features + ['Case of flush']]
A2 = Z2.corr()


# VIF 3
X3 = df["Red LED 2\nPhotodiode 2"]
y3 = df["Green LED 2\nPhotodiode 2"]
model = sm.OLS(y1, X1)
results = model.fit()
res_GL2_2 = results.resid
df["res_GL2_2"] = res_GL2_2  # Ajoutez cette ligne pour créer la nouvelle colonne

features = ["Blue LED 1\nPhotodiode 1", "Blue LED 1\nPhotodiode 2",
           "res_GL1_1", "Green LED 1\nPhotodiode 2", 
           "Red LED 1\nPhotodiode 1", "Red LED 1\nPhotodiode 2", 
           "Blue LED 2\nPhotodiode 1", "Blue LED 2\nPhotodiode 2", 
           "res_GL2_1", "res_GL2_2", 
           "Red LED 2\nPhotodiode 1", "Red LED 2\nPhotodiode 2"]
X = df[features]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

Z3 = df[features + ['Case of flush']]
A3 = Z3.corr()


# VIF 4
X4 = df["Red LED 1\nPhotodiode 2"]
y4 = df["Green LED 1\nPhotodiode 2"]
model = sm.OLS(y1, X1)
results = model.fit()
res_GL1_2 = results.resid
df["res_GL1_2"] = res_GL1_2  # Ajoutez cette ligne pour créer la nouvelle colonne

features = ["Blue LED 1\nPhotodiode 1", "Blue LED 1\nPhotodiode 2",
           "res_GL1_1", "res_GL1_2", 
           "Red LED 1\nPhotodiode 1", "Red LED 1\nPhotodiode 2", 
           "Blue LED 2\nPhotodiode 1", "Blue LED 2\nPhotodiode 2", 
           "res_GL2_1", "res_GL2_2", 
           "Red LED 2\nPhotodiode 1", "Red LED 2\nPhotodiode 2"]


X = df[features]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

df.to_excel('mon_fichier.xlsx', index=False)



# Pairplot exploration

sns.pairplot(X)


sns.displot(df, x="Red LED 1\nPhotodiode 2", hue="Urine level", palette="flare")
plt.show()

sns.scatterplot(df, x= 'Blue LED 1\nPhotodiode 2', y='Red LED 1\nPhotodiode 2', hue="Feces Level",  palette="flare")
plt.show()

sns.scatterplot(df, x= 'Blue LED 1\nPhotodiode 1', y='Green LED 1\nPhotodiode 1', hue="Urine level",  palette="flare")
plt.show()

sns.scatterplot(df, x= 'Red LED 2\nPhotodiode 2', y='Red LED 1\nPhotodiode 2', hue="Feces Level",  palette="flare")
plt.show()

sns.scatterplot(df, x= 'Blue LED 1\nPhotodiode 1', y= 'Blue LED 1\nPhotodiode 2')

sns.scatterplot(df, x= 'Red LED 2\nPhotodiode 2', y='Blue LED 2\nPhotodiode 2', hue="Urine level")
plt.show()

sns.scatterplot(df, x= 'Case of flush', y='Blue LED 2\nPhotodiode 2')
plt.show()



# Analysis Discrete/conitnuous 

fig, ax = plt.subplots(len(features) // 3, 3, figsize=(14, 9))
ax = ax.flatten()
for i in range(len(features)):
    sns.boxplot(df, x="Case of flush", y=features[i], ax=ax[i])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(len(features) // 3, 3, figsize=(14, 9))
ax = ax.flatten()
for i in range(len(features)):
    sns.boxplot(df, x="Urine level", y=features[i], ax=ax[i])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(len(features) // 3, 3, figsize=(14, 9))
ax = ax.flatten()
for i in range(len(features)):
    sns.boxplot(df, x="Feces Level", y=features[i], ax=ax[i])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(len(features) // 3, 3, figsize=(14, 9))
ax = ax.flatten()
for i in range(len(features)):
    sns.boxplot(df, x="Paper level", y=features[i], ax=ax[i])
plt.tight_layout()
plt.show()


