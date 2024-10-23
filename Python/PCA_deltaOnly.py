# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:25:39 2024

@author: Yitong Iris Shen 
"""

"""
PCA for FRAP, 8 variables (delta only)
"""

import os
import platform
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity, FactorAnalyzer
import seaborn as sns
from scipy.stats import zscore, pearsonr


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

# Making sure all vaues are numeric values 
print(deltadata.dtypes) #both integer and float are numeric types, so it will work fine for PCA

# Clean up NaN (missing values) for PCA purpose
deltadata_noNA = deltadata.dropna()
deltadata_noNA .info()
participant_ids = deltadata_noNA['SubID'].tolist()
print(participant_ids) #[16, 17, 19, 24, 26, 27, 30, 34, 35, 40, 44, 71, 80, 93, 96, 100], n=16
#Checking for missing values to ensure
print(deltadata_noNA.isna().sum())

# Removing outliers
# Define a threshold for Z-scores
threshold = 3

# Create a copy of the dataset for removing outliers
deltadata_no_outliers = deltadata_noNA.copy()

# Loop through each DD variable and identify outliers greater than the threshold
for col in deltadata_noNA:
    # Calculate the Z-score for the variable
    z_scores = np.abs(zscore(deltadata_noNA[col]))

    # Identify outliers using the threshold
    outliers = z_scores > threshold

    # Remove the outliers from the dataset (only keep rows where outliers is False)
    deltadata_no_outliers = deltadata_no_outliers[~outliers]

####################################### PCA #######################################
# Selecting variables I want to include 
selected_columns = ['Delta_ConfidenceRecovery', 'Delta_LikelyRelapse30Days', 'Delta_LikelyRelapse1yr',
                    'Delta_RecoveryImportance', 'Delta_Craving', 'Delta_FutSim', 'Delta_FutConn',
                    'Delta_AUC', 'DD_IChoRat']
PCA_9variables_data = deltadata_clean[selected_columns]
PCA_9variables_data.to_csv(r'C:\Users\shen21\Desktop\PCA\Python\PCA_9variables_data.csv', index=False)

# Step 0: Performing Kaiser-Meyer-Olkin (KMO) Test and Bartlett’s Test of Sphericity
# KMO
kmo_all, kmo_model = calculate_kmo(PCA_9variables_data)
print(f"KMO Score: {kmo_model}") #KMO Score: 0.48112756149640323
# Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(PCA_9variables_data)
print(f"Bartlett's Test: Chi-Square = {chi_square_value}, p-value = {p_value}") #Chi-Square = 22.388860370116944, p = 0.9629457075693935

# Step 1: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(PCA_9variables_data)

# Convert the scaled data back to a DataFrame
scaled_data_df = pd.DataFrame(scaled_data, columns=selected_columns)

# Step 2: Perform PCA to determine the number of components with eigenvalue > 1
pca = PCA()
pca.fit(scaled_data_df)

# Extract components with eigenvalues > 1
eigenvalues = pca.explained_variance_
num_components = sum(eigenvalues > 1)
print(f"Number of components with eigenvalue > 1: {num_components}")

# Step 3: Perform Factor Analysis with Varimax rotation for the determined number of components
fa = FactorAnalyzer(n_factors=num_components, rotation='varimax')
fa.fit(scaled_data_df)

# Get the rotated loadings
rotated_loadings = fa.loadings_
loadings_df_rotated = pd.DataFrame(rotated_loadings, columns=[f'PC{i+1}' for i in range(num_components)], index=selected_columns)

# Print the rotated loadings
print(loadings_df_rotated)

# Create a heatmap for the rotated loadings
plt.figure(figsize=(10, 8))
ax = sns.heatmap(loadings_df_rotated, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=.5)

# Bold values above a threshold
for text in ax.texts:
    value = float(text.get_text())
    if abs(value) > 0.4:
        text.set_weight('bold')

plt.title('PCA Loadings Heatmap with Varimax Rotation (Aligned with SPSS Settings)')
plt.xlabel('Principal Components')
plt.ylabel('Variables')
plt.show()