# -*- coding: utf-8 -*-
"""
PCA for FRAP
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset (FRAP), note we are using the csv file as the cleaned up data
file_path = 'K:\\PSYC\\Oberlin_Lab\\2. Projects, Collaborations, and Grants\\0. Projects\\Iris\\FRAP neuroimaging paper\\Python'
data = pd.read_excel('Data_10182024.xlsx')
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
output_file_path = 'K:\\PSYC\\Oberlin_Lab\\2. Projects, Collaborations, and Grants\\0. Projects\\Iris\\FRAP neuroimaging paper\\Python\\Data_10182024_with_deltaVR.xlsx'
data_with_deltaVR.to_excel(output_file_path, index=False)
print(f"Data with deltas saved to: {output_file_path}")

# Read updated data
# Absolute path to the file when using my Mac
data_path_macbook = '/Users/irisshen/Desktop/PCA/Python/Data_10182024_with_deltaVR.xlsx'

# Check if the file exists 
if os.path.exists(data_path_macbook):
    print("File exists!")
else:
    print("File does not exist at the absolute path.")

# Load the Excel file into a pandas DataFrame
deltadata = pd.read_excel(data_path_macbook)
# Checking my variables 
print(deltadata.columns)

####################################### PCA #######################################
# Selecting variables I want to include 
selected_columns = ['Delta_ConfidenceRecovery', 'Delta_LikelyRelapse30Days', 'Delta_LikelyRelapse1yr',
                    'Delta_RecoveryImportance', 'Delta_Craving', 'Delta_FutSim', 'Delta_FutConn',
                    'Delta_AUC', 'DD_IChoRat']

# Step 1: Select only numeric features for PCA
features = deltadata.select_dtypes(include=[float, int]).columns
# Step 2: Standardize the data
scaler = StandardScaler()
scaled_deltadata = scaler.fit_transform(deltadata[features])

# Step 3: Fit PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Step 4: Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Show the explained variance ratio for each component
print("Explained variance by each component:", pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()