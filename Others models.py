#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 06:07:58 2023

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline




# 1. Load the dataset
file_path = 'C:/Users/alec-/Working directory/project/mon_fichier.xlsx'
data = pd.read_excel(file_path)



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

# 4. Preprocessor
preprocessor = make_pipeline(PolynomialFeatures(2), SelectKBest(f_classif, k=10))

# 5. Models
model_SVC = make_pipeline(preprocessor, StandardScaler(), SVC())
model_SGDC = make_pipeline(preprocessor, StandardScaler(), SGDClassifier(random_state=0))

# 6. List of models
list_of_models = {
    'SVC': model_SVC, 
    'SGDClassifier': model_SGDC, 
}

# 7. Training and evaluating each model
for name, model in list_of_models.items():
    print(f"Training and evaluating {name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_train).round()
    print(classification_report(y_train, predictions))
    predictions = model.predict(X_test).round()
    print(classification_report(y_test, predictions))
    
##################################################################################################
######################################    Optimization      ######################################
##################################################################################################

# 8. SVC Optimisation (no SGDClassifier because poorest result)

param_grid_svc = {
    'pipeline__polynomialfeatures__degree': [1, 2, 3],    
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma': ['scale', 'auto']
}

model_SVC = Pipeline([
    ('pipeline', preprocessor),
    ('standardscaler', StandardScaler()),
    ('svc', SVC())
])

grid_search_svc = GridSearchCV(model_SVC, param_grid_svc, cv=5, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)

print("Best parameters:", grid_search_svc.best_params_)
best_model = grid_search_svc.best_estimator_

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
print("safe flush accuracy test:", get_safe_flush_accuracy(y_test, predictions_test))


##################################################################################################
######################################    Water Focus    #########################################
##################################################################################################

print("Mean Absolute Error train:", mean_absolute_error(y_train, predictions_train))
print("Mean Absolute Error test:", mean_absolute_error(y_test, predictions_test))

