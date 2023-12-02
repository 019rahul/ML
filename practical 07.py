import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create a DataFrame from your custom dataset
data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Location': ['New York City', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Philadelphia', 'Phoenix', 'Dallas', 'San Diego', 'San Francisco'],
    'Arrest Date': ['2023-01-05', '2023-01-10', '2023-02-15', '2023-03-20', '2023-04-25', '2023-05-01', '2023-06-10', '2023-07-15', '2023-08-20', '2023-09-05'],
    'Offense': ['Assault', 'Drug Possession', 'Burglary', 'DUI', 'Robbery', 'Assault', 'Drug Trafficking', 'Theft', 'Domestic Violence', 'Public Intoxication'],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Male'],
    'Age': [27, 34, 42, 29, 22, 31, 38, 35, 28, 24],
    'ArrestingOfficer': ['Officer Smith', 'Officer Johnson', 'Officer Brown', 'Officer Davis', 'Officer Wilson', 'Officer Martinez', 'Officer Anderson', 'Officer Thomas', 'Officer Garcia', 'Officer Rodriguez']
})

# Select the 'Age' column for PCA
X = data[['Age']].values

# Standardize the feature matrix (mean=0, variance=1)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

n_components = 1
pca = PCA(n_components=n_components)

# Fit and transform the data to the first 'n_components' principal components
X_pca = pca.fit_transform(X_std)

# Create a DataFrame to store the results
result = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

# Add the PCA results to the original DataFrame
data['PCA'] = result['PC1']

# Plot the data along the first principal component
plt.scatter(data['PCA'], np.zeros_like(data['PCA']), alpha=0.5)
plt.xlabel('Principal Component 1')
plt.title('Data along Principal Component 1')


# Display the resulting DataFrame with PCA results.
print(data)

#print the graph
plt.show()
