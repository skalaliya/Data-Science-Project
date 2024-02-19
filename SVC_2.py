#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:26:53 2023

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter




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
model_SVC = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
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
    'svc__kernel': ['linear', 'poly'], # We cannot we the RBF kernel, because this one has a potential infinite number of dimensions (against the mission instructions)
    'svc__gamma': ['scale', 'auto']
}

model_SVC = Pipeline([
    ('pipeline', preprocessor),
    ('standardscaler', StandardScaler()),
    ('svc', SVC(probability=True))  # Set probability to True here
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
######################################    Case Adaptation    #####################################
##################################################################################################

# 1. Property metrics
def get_safe_flush_accuracy(y_true, y_pred):
    return np.mean(y_pred >= y_true)


predictions_train_2 = best_model.predict_proba(X_train)
print(predictions_train_2.round(3))

pd.Series(predictions_train_2[304]).plot.bar()
print(y_train.iloc[304])

# 2. Define the threshold
threshold = 0.1

# 3. Train_set

# Initialize an array to hold the cutoff class for each sample
cutoff_classes_train = np.zeros(predictions_train_2.shape[0], dtype=int)

# Iterate over each sample's predictions
for i, probs in enumerate(predictions_train_2):
    reversed_probs = probs[::-1]
    cutoff_index = np.argmax(reversed_probs > threshold)
    cutoff_class = max(0, len(reversed_probs) - 1 - cutoff_index)
    cutoff_classes_train[i] = cutoff_class

# Using the get_safe_flush_accuracy function with y_test and cutoff_classes
safe_flush_accuracy_train = get_safe_flush_accuracy(y_train, cutoff_classes_train)

# 4. Test_set

predictions_test_2 = best_model.predict_proba(X_test)
print(predictions_test_2.round(3))

pd.Series(predictions_test_2[195]).plot.bar()
print(y_test.iloc[195])

cutoff_classes_test = np.zeros(predictions_test_2.shape[0], dtype=int)

for i, probs in enumerate(predictions_test_2):
    reversed_probs = probs[::-1]
    cutoff_index = np.argmax(reversed_probs > threshold)
    cutoff_class = max(0, len(reversed_probs) - 1 - cutoff_index)
    cutoff_classes_test[i] = cutoff_class

safe_flush_accuracy_test = get_safe_flush_accuracy(y_test, cutoff_classes_test)

# 5. Cleanliness concerns
print("Safe Flush Accuracy train:", safe_flush_accuracy_train)
print("Safe Flush Accuracy test:", safe_flush_accuracy_test)

# 6. Water concerns
print("MAE train", mean_absolute_error(y_train, cutoff_classes_train))
print("MAE test:", mean_absolute_error(y_test, cutoff_classes_test))



predictions_test = model.predict(X_test).round()

# Now use Counter to count the occurrences of each predicted level
level_counts = Counter(predictions_test)

# This will give you a dictionary with the level as the key and the count as the value
print(level_counts)
flush_level_to_water_volume = {1: 1.5, 2: 1.9, 3: 2.4, 4: 2.8, 5: 3.3, 6: 3.8, 7: 4.2, 8: 4.7, 9: 5.2, 10: 5.6, 11: 6.1
                              }

# Assuming 'level_counts' contains the count of each predicted flush level
total_volume = 0
total_predictions = 0

for level, count in level_counts.items():
    volume = flush_level_to_water_volume.get(level, 0)
    total_volume += volume * count
    total_predictions += count

# Calculate the average flush volume
if total_predictions > 0:
    average_flush_volume = total_volume / total_predictions
    print("Average Flush Volume:", average_flush_volume)
else:
    print("No predictions to calculate average flush volume")

# Constants
water_price_per_thousand_liters = 4  # Price in euros
rooms = 100  # Number of rooms in the hotel
flushes_per_room_per_day = 5  # Number of flushes per room per day

classical_volume_per_flush = 6  # Classical system water usage in liters
savings_per_flush = classical_volume_per_flush - average_flush_volume

water_cost_per_liter = water_price_per_thousand_liters / 1000  # Convert price to cost per liter
daily_cost_savings = savings_per_flush * water_cost_per_liter * flushes_per_room_per_day * rooms
annual_cost_savings = daily_cost_savings * 365

annual_water_savings = savings_per_flush * flushes_per_room_per_day * rooms * 365

# Print the results
print(f"Average Volume Per Flush: {average_flush_volume} liters")
print(f"Savings Per Flush: {savings_per_flush} liters")
print(f"Daily Cost Savings: €{daily_cost_savings}")
print(f"Annual Cost Savings: €{annual_cost_savings}")
print(f"Annual Water Savings: {annual_water_savings} liters")

