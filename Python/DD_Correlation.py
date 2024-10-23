# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:07:16 2024

@author: shen21

The method we used in the Delay Discounting (DD) task in the scanner is non-adjusting while all other DD tasks done outside 
of the scanner were adjusting for indifference point. Wanting to check if the choice ratio for the task in the scanner is 
correlated with other measures of DD.
"""

import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore, pearsonr
import seaborn as sns

########################### Define paths for MacBook and work desktop ###########################
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
    
########################### Identifying outliers for all DD variables ###########################
# Extract all DD columns and "AUC" columns
dd_columns = [col for col in deltadata.columns if "DD" in col or "AUC" in col]
print("DD Variables:", dd_columns)

# Define a threshold for Z-scores
threshold = 3

# Create a copy of the dataset for removing outliers
deltadata_no_outliers = deltadata.copy()

# Loop through each DD variable and identify outliers greater than the threshold
for col in dd_columns:
    # Calculate the Z-score for the variable
    z_scores = np.abs(zscore(deltadata[col]))

    # Identify outliers using the threshold
    outliers = z_scores > threshold

    # Remove the outliers from the dataset (only keep rows where outliers is False)
    deltadata_no_outliers = deltadata_no_outliers[~outliers]

# Step 2: Filter to include only the columns of interest (DD and AUC columns)
filtered_data = deltadata_no_outliers[dd_columns]

# Drop rows with NaN or inf values
filtered_data_clean = filtered_data.replace([np.inf, -np.inf], np.nan).dropna()

# Ensure filtered_data_clean is not empty
if filtered_data_clean.empty:
    print("Error: No data left after removing outliers.")
else:
    # Initialize empty DataFrames to store the correlation coefficients and p-values
    correlation_coefficients = pd.DataFrame(index=filtered_data_clean.columns, columns=filtered_data_clean.columns)
    p_values = pd.DataFrame(index=filtered_data_clean.columns, columns=filtered_data_clean.columns)

    # Loop through each pair of columns to calculate Pearson correlation and p-value
    for col1 in filtered_data_clean.columns:
        for col2 in filtered_data_clean.columns:
            if col1 == col2:
                # The correlation of a variable with itself is 1, and p-value is NaN
                correlation_coefficients.loc[col1, col2] = 1.0
                p_values.loc[col1, col2] = np.nan
            else:
                # Calculate Pearson correlation and p-value
                corr_coeff, p_value = pearsonr(filtered_data_clean[col1], filtered_data_clean[col2])
                correlation_coefficients.loc[col1, col2] = corr_coeff
                p_values.loc[col1, col2] = p_value

    # Convert the DataFrames to numeric types (to ensure they are properly formatted for plotting)
    correlation_coefficients = correlation_coefficients.astype(float)
    p_values = p_values.astype(float)

    # Print the correlation coefficients and p-values DataFrames
    print("Correlation Coefficients:")
    print(correlation_coefficients)
    print("\nP-values:")
    print(p_values)

    # Option 1: Create Separate Heatmaps
    plt.figure(figsize=(14, 6))

    # Heatmap for Correlation Coefficients
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_coefficients, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Correlation Coefficients Heatmap (Without Outliers)')

    # Heatmap for P-values
    plt.subplot(1, 2, 2)
    sns.heatmap(p_values, annot=True, cmap='viridis', center=0.05, linewidths=0.5)
    plt.title('P-values Heatmap (Without Outliers)')

    plt.tight_layout()
    plt.show()

    # Option 2: Create a Combined Table of Correlation Coefficients and P-values
    combined_df = pd.DataFrame(index=filtered_data_clean.columns, columns=filtered_data_clean.columns)

    # Combine correlation coefficient and p-value into a string format
    for col1 in filtered_data_clean.columns:
        for col2 in filtered_data_clean.columns:
            coeff = correlation_coefficients.loc[col1, col2]
            p_val = p_values.loc[col1, col2]
            combined_df.loc[col1, col2] = f"{coeff:.2f} (p={p_val:.2g})"

    # Print the combined DataFrame
    print("Combined Correlation Coefficients and P-values:")
    print(combined_df)
    
# Scatter plot to visualize the relationship between DD Immediate Choice ratio and Delta DD AUC 
# (we expected them to be negatively correlated but they mapped on the same direction in PCA)
# Ensure the two columns exist in the DataFrame
if 'Delta_AUC' in filtered_data_clean and 'DD_IChoRat' in filtered_data_clean:
    
    # Step 1: Extract the data for the two variables
    x = filtered_data_clean['Delta_AUC']
    y = filtered_data_clean['DD_IChoRat']
    
    # Step 2: Create the scatter plot
    plt.figure(figsize=(8,6))  # Optional: specify the size of the plot
    plt.scatter(x, y, c='blue', alpha=0.5)
    
    # Step 3: Add labels and title
    plt.title('Scatter Plot: Delta_DD_AUC vs DD_IChoRat')
    plt.xlabel('Delta_DD_AUC')
    plt.ylabel('DD_IChoRat')
    
    # Step 4: Display the plot
    plt.grid(True)
    plt.show()
else:
    print("Columns 'Delta_DD_AUC' and/or 'DD_IChoRat' not found in the DataFrame.")
    