#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:39:22 2023

@author: arnaudcruchaudet
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the dataset
file_path = 'C:/Users/alec-/Working directory/project/mon_fichier.xlsx'
data = pd.read_excel(file_path)


# 2. Define features and target
features = ["Blue LED 1\nPhotodiode 1", "Blue LED 1\nPhotodiode 2",
           "res_GL1_1", "res_GL1_2", 
           "Red LED 1\nPhotodiode 1", "Red LED 1\nPhotodiode 2", 
           "Blue LED 2\nPhotodiode 1", "Blue LED 2\nPhotodiode 2", 
           "res_GL2_1", "res_GL2_2", 
           "Red LED 2\nPhotodiode 1", "Red LED 2\nPhotodiode 2"]
X = data[features]
y = data['Case of flush']

# 3. Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



##################################################################################################
######################################    Modellisation     ######################################
##################################################################################################

model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

predictions = model.predict(X_train).round()
print(classification_report(y_train, predictions))

predictions = model.predict(X_test).round()
print(classification_report(y_test, predictions))


##################################################################################################
######################################    Optimization      ######################################
##################################################################################################

# WE PUT THE GRID_SEARCH IN COMMENT TO AVOID TO RUN IT EACH TIME
# OF COURSE, IT WAS BEFORE WE DISCOVER THE RANDOM_STATE PARAMS
#param_grid = {
    #'polynomialfeatures__degree': [1, 2, 3],              # Degrees of the polynomial features
    #'logisticregression__C': [0.1, 1, 10, 100, 300],          # Regularization strength
    #'logisticregression__penalty': ['l1', 'l2'],               # Type of penalty (l1 not supported with multinomial)
    #'logisticregression__multi_class': ['auto', 'ovr', 'multinomial'],  # Type of multi-class approach
    #'logisticregression__solver': ['lbfgs', 'sag', 'saga', 'newton-cg'], # Solvers that support multinomial
    #'logisticregression__max_iter': [100, 500, 1000, 2000, 3000]  # Max iteration number
# }

#grid_search = GridSearchCV(
    #make_pipeline(PolynomialFeatures(), StandardScaler(), LogisticRegression()),
    #param_grid,
    #cv=5,  
    #scoring='accuracy',
    #verbose=1
#)
#grid_search.fit(X_train, y_train)

#print("Best parameters:", grid_search.best_params_)
#best_model = grid_search.best_estimator_

best_model = make_pipeline(
    PolynomialFeatures(degree=3), # The best degree for PolynomialFeatures
    StandardScaler(),
    LogisticRegression(
        C=10, # Regularization strength
        max_iter=100, # Max iteration number
        penalty='l2', # Type of penalty
        multi_class='ovr', # Type of multi-class approach
        solver='lbfgs' # Solver that supports the chosen multi_class option
    )
)

# 4. Fit the model with the training data
best_model.fit(X_train, y_train)



test_predictions = best_model.predict(X_test)

predictions_train = best_model.predict(X_train)
print(classification_report(y_train, predictions_train))

predictions_test = best_model.predict(X_test)
print(classification_report(y_test, predictions_test))


##################################################################################################
######################################    Cleanliness Focus    ######################################
##################################################################################################

def get_safe_flush_accuracy(y_true, y_pred):
    return np.mean(y_pred >= y_true)


predictions_train += 1
print("safe flush accuracy train:", get_safe_flush_accuracy(y_train, predictions_train))

predictions_test += 1
print("safe flush accuracy test;", get_safe_flush_accuracy(y_test, predictions_test))

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
sns.heatmap(confusion_matrix(y_test, predictions_test), annot=True, ax=ax)
plt.show()
##################################################################################################
######################################    Water Focus    #########################################
##################################################################################################

print("Mean Absolute Error train:", mean_absolute_error(y_train, predictions_train))
print("Mean Absolute Error test:", mean_absolute_error(y_test, predictions_test))

























    

