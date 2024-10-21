# -*- coding: utf-8 -*-
"""
PCA for FRAP
"""

import os
import platform
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Load dataset (FRAP), note we are using the csv file as the cleaned up data
file_path = r'C:\Users\shen21\Desktop\PCA\Python\Data_10182024.xlsx'
if os.path.exists(file_path):
    print(f"File found at: {file_path}")
    # Load the Excel file into a pandas DataFrame
    data = pd.read_excel(file_path)
    # Print the first few rows of the data
    print(data.head())
else:
    print(f"File not found: {file_path}")
    print(data.head())

#################### Create a new DataFrame for the differences ####################
# Preprocessing the data and calculate difference scores
# Define the columns for the two time points
prevr_columns = ['PreVR_MS2_ConfidenceRecovery', 'PreVR_MS2_LikelyRelapse30Days', 'PreVR_MS2_LikelyRelapse1yr', 
                 'PreVR_MS2_RecoveryImportance', 'PreVR_MS2_Craving', 'PreVR_MS2_FutSim', 'PreVR_MS2_FutConn', 'DD1_AUC']
postvr_columns = ['PostVR_MS3_ConfidenceRecovery', 'PostVR_MS3_LikelyRelapse30Days', 'PostVR_MS3_LikelyRelapse1yr', 
                 'PostVR_MS3_RecoveryImportance', 'PostVR_MS3_Craving', 'PostVR_MS3_FutSim', 'PostVR_MS3_FutConn', 'DD2_AUC']

# Calculate the difference (PostVR - PreVR)
delta = data[postvr_columns].values - data[prevr_columns].values

# function splits the string wherever an underscore (_) appears, 
# and it produces a list where each part of the string (separated by underscores) becomes an element. We want the last element so -1
delta_columns = [f'Delta_{col.split("_")[-1]}' for col in prevr_columns]
delta_PrePostVR = pd.DataFrame(delta, columns=delta_columns)

# Add new variables to original data and export it
data_with_deltaVR = pd.concat([data, delta_PrePostVR], axis=1)
# Specify the output file path
output_file_path = r'C:\Users\shen21\Desktop\PCA\Python\Data_10182024_withDeltaVR.xlsx'
# Save the updated DataFrame to Excel
data_with_deltaVR.to_excel(output_file_path, index=False)
print(f"Data with deltas saved to: {output_file_path}")

# Read updated data
# Define paths for MacBook and work desktop
data_path_macbook = '/Users/irisshen/Desktop/PCA/Python/Data_10182024_withDeltaVR.xlsx'
data_path_work = 'C:\\Users\\shen21\\Desktop\\PCA\\Python\\Data_10182024_withDeltaVR.xlsx'

# Check the current operating system
if platform.system() == "Darwin":  # Darwin is the system name for macOS
    data_path = data_path_macbook
    print("Running on MacBook, using macOS path")
elif platform.system() == "Windows":
    data_path = data_path_work
    print("Running on work desktop, using Windows path")
else:
    raise Exception("Unsupported platform")

# Check if the file exists at the chosen path
if os.path.exists(data_path):
    print("File exists!")
else:
    raise FileNotFoundError(f"File does not exist at the path: {data_path}")

# Load the Excel file into a pandas DataFrame
deltadata = pd.read_excel(data_path)

# Checking the variables (columns)
print(deltadata.columns)

####################################### PCA #######################################
# Selecting variables I want to include 
selected_columns = ['Delta_ConfidenceRecovery', 'Delta_LikelyRelapse30Days', 'Delta_LikelyRelapse1yr',
                    'Delta_RecoveryImportance', 'Delta_Craving', 'Delta_FutSim', 'Delta_FutConn',
                    'Delta_AUC', 'DD_IChoRat']
# Clean up NaN (missing values)
deltadata_clean = deltadata.dropna()
deltadata_clean.info()
participant_ids = deltadata_clean['SubID'].tolist()
print(participant_ids) #[16, 17, 19, 24, 26, 27, 30, 34, 35, 40, 44, 71, 80, 93, 96, 100], n=16

# Step 1: Select only numeric features for PCA
features = deltadata_clean.select_dtypes(include=[float, int]).columns

# Step 2: Standardize the data
scaler = StandardScaler()
scaled_deltadata_clean = scaler.fit_transform(deltadata_clean[features])

# Step 3: Performing Kaiser-Meyer-Olkin (KMO) Test and Bartlett’s Test of Sphericity
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
kmo_all, kmo_model = calculate_kmo(deltadata_clean)
print(f"KMO Score: {kmo_model}")

# Perform Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(deltadata_clean)
print(f"Bartlett's Test: Chi-Square = {chi_square_value}, p-value = {p_value}")

# Assuming 'data' is your DataFrame
data = pd.read_excel('Data_10182024.xlsx')

# KMO Test
kmo_all, kmo_model = calculate_kmo(data)
print(f"KMO Score: {kmo_model}")

# Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(data)
print(f"Bartlett's Test: Chi-Square = {chi_square_value}, p-value = {p_value}")

pca = PCA()
pca.fit(scaled_deltadata_clean)

# Eigenvalues (explained variance for each component)
eigenvalues = pca.explained_variance_

# Keep components with eigenvalues > 1
n_components_kaiser = sum(eigenvalues > 1)

print(f"Number of components using Kaiser's criterion: {n_components_kaiser}")

plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
plt.title('Scree Plot')
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue (Explained Variance)')
plt.axhline(y=1, color='r', linestyle='--')  # Kaiser criterion line
plt.show()