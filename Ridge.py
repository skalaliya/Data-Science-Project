#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:29:16 2023

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from collections import Counter


# OPEN DATA
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##################################################################################################
######################################    Modellisation     ######################################
##################################################################################################

model = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(random_state=0))
model.fit(X_train, y_train)

predictions = model.predict(X_train).round()
print(classification_report(y_train, predictions))

predictions = model.predict(X_test).round()
print(classification_report(y_test, predictions))


##################################################################################################
######################################    Optimization      ######################################
##################################################################################################

param_grid = {
    'polynomialfeatures__degree': [1, 2, 3],
    'ridge__alpha': [0.1, 0.3, 1, 3, 10, 30, 100], 
}


grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate the model
predictions_train = best_model.predict(X_train).round()
print(classification_report(y_train, predictions_train))

predictions_test = best_model.predict(X_test).round()
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

fig, ax = plt.subplots(1, 1, figsize=(7, 3))
sns.heatmap(confusion_matrix(y_test, predictions_test), annot=True, ax=ax)
plt.show()
##################################################################################################
######################################    Water Focus    #########################################
##################################################################################################

print("Mean Absolute Error train:", mean_absolute_error(y_train, predictions_train))
print("Mean Absolute Error test:", mean_absolute_error(y_test, predictions_test))


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


    
    
    
