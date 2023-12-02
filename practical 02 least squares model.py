import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  # Corrected the import for r2_score

# Generate a synthetic college dataset (replace this with your actual dataset)
np.random.seed(0)
X = np.random.rand(100, 1)  # Input feature (e.g., college admission scores)
y = 2 * X + 1 + np.random.randn(100, 1)  # Output/target variable (e.g., college GPA)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model using least squares
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("\nMean Squared Error:", mse)
print("\nR-squared:", r2)

# Print the model coefficients (slope and intercept)
print("\nModel Coefficients (slope and intercept):", model.coef_, model.intercept_)
