import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

# Suppress the Polynomial Features warning
warnings.filterwarnings("ignore", category=UserWarning)

# Create a synthetic carseats dataset
np.random.seed(0)
n_samples = 100
data = {
    'Sales': np.random.randint(50, 400, n_samples),
    'Price': np.random.uniform(50, 300, n_samples),
    'Advertising': np.random.randint(1, 20, n_samples),
    'CompPrice': np.random.uniform(80, 350, n_samples),
    'Income': np.random.randint(20, 80, n_samples),
    'Population': np.random.randint(100, 1000, n_samples),
}

# Create a DataFrame
carseats_df = pd.DataFrame(data)

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(carseats_df.head())

# Perform polynomial regression
X = carseats_df[['Price']]
y = carseats_df['Sales']
degree = 2

# Transform the features into polynomial features
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the polynomial features
model.fit(X_poly, y)

# Predict on the same X values
y_pred = model.predict(X_poly)

# Plot the original data points
plt.scatter(X, y, label="Original Data")

# Sort the X values for a smoother curve in the plot
X_sorted = np.sort(X, axis=0)
y_pred_sorted = model.predict(poly_features.transform(X_sorted))

# Plot the polynomial fit line
plt.plot(X_sorted, y_pred_sorted, color='red', label="Polynomial Fit")

# Add labels and legend
plt.xlabel("Price")
plt.ylabel("Sales")
plt.legend()

# Get the coefficients of the polynomial regression model
coefficients = model.coef_

# Print the coefficients
print("\nPolynomial Regression Coefficients:")
for i, coef in enumerate(coefficients):
    print(f"Coef({i}): {coef}")

# Print the intercept
intercept = model.intercept_
print(f"Intercept: {intercept}")

# Show the plot
plt.show()
