import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Load the college dataset from the provided data
data = pd.DataFrame({
    'University Name': ['Harvard University', 'Stanford University', 'MIT', 'Caltech', 'Princeton', 'Yale', 'Columbia', 'UChicago'],
    'Enrollment': [21000, 17000, 11000, 2200, 14000, 13000, 29000, 16000],
    'Acceptance Rate': ['5', '5', '17', '5', '6', '65', '74', ''],
    'Student-Faculty Ratio': ['7:1', '5:1', '3:1', '3:1', '15:1', '16:1', '6:1', '5:1'],
    'Tuition': [45000, 47000, 49000, 50000, 48000, 46000, 49000, 47000],
    'Graduation Rate': ['95%', '9', '93', '91', '96', '96', '94%', '92'],
    'Median SAT Score': [1520, 1510, 1570, 1590, 1490, 1500, 1530, 1480]
})

# Replace empty strings with a default value (e.g., -1)
data['Acceptance Rate'] = data['Acceptance Rate'].replace('', '-1')

# Convert columns with percentage values to numeric
data['Acceptance Rate'] = data['Acceptance Rate'].str.replace('%', '').astype(float)
data['Graduation Rate'] = data['Graduation Rate'].str.replace('%', '').astype(float)

# Process the 'Student-Faculty Ratio' column to convert it to a numerical format
data['Student-Faculty Ratio'] = data['Student-Faculty Ratio'].apply(lambda x: float(x.split(':')[0]) / float(x.split(':')[1]))

# Split the data into features (X) and the target variable (y)
X = data.drop(['University Name', 'Enrollment'], axis=1)  # Excluding non-numeric columns
y = data['Enrollment']  # Predicting enrollment

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a Lasso regression model with a higher max_iter value
lasso_model = Lasso(alpha=1.0, max_iter=10000)  # Increase max_iter to 10000 or higher
lasso_model.fit(X_train, y_train)

# Predict on the test data
y_pred = lasso_model.predict(X_test)

# Evaluate the model's performance (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
