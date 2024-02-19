#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 03:54:47 2023

@author: arnaudcruchaudet
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from collections import Counter


# 1. Load the dataset
file_path = 'C:/Users/alec-/Working directory/project/mon_fichier.xlsx'
data = pd.read_excel(file_path)


data.info()
X = data['PCB value'] # Predicted values
Y = data['Case of flush'] # Actual value
 
matrix = confusion_matrix(Y, X)

sns.heatmap(matrix, annot=True, fmt="d")
plt.ylabel('Actual Value')
plt.xlabel('Predicted Value')
plt.show()

print(classification_report(X, Y))

def get_safe_flush_accuracy(y_true, y_pred):
    return np.mean(y_pred >= y_true)
 
competitor_safe_flush_accuracy = get_safe_flush_accuracy(Y, X)

competitor_mae = mean_absolute_error(Y, X)

# Print the results
print(f"Competitor's Safe Flush Accuracy: {competitor_safe_flush_accuracy:.2f}")
print(f"Competitor's Mean Abslute Error: {competitor_mae:.2f}")
 


# Extracting the "PCB value" column
pcb_values = data['PCB value']

# Counting the occurrences of each number in the "PCB value" column
pcb_value_counts = Counter(pcb_values)

# Printing the counts
for value, count in pcb_value_counts.items():
    print(f"PCB Value {value}: {count} occurrences")


    flush_level_to_water_volume = {1: 1.5, 2: 1.9, 3: 2.4, 4: 2.8, 5: 3.3, 6: 3.8, 7: 4.2, 8: 4.7, 9: 5.2, 10: 5.6, 11: 6.1
                                  }

# Step 1: Multiply count of each PCB value by its corresponding water volume
total_volume = sum(pcb_value_counts[pcb_value] * flush_level_to_water_volume.get(pcb_value, 0)
                   for pcb_value in pcb_value_counts)
print("Total Volume:", total_volume)
# Step 2: Sum of all these products (already done in the line above)

# Step 3: Divide by the total number of observations (976)
average_flush_volume = total_volume / 976

print("Average Volume:", average_flush_volume)

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
