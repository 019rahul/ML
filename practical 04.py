import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Create a DataFrame with the provided example data
data = pd.DataFrame({
    'Sales': [9.5, 11.22, 10.06, 7.41],
    'Income': [73, 48, 35, 100],
    'CompPrice': [138, 111, 113, 1171],
    'Advertising': [11, 16, 10, 41],
    'ShelveLoc': ['Bad', 'Good', 'Medium', 'Medium'],
    'Education': [17, 10, 12, 14],
    'Urban': ['Yes', 'Yes', 'Yes', 'Yes'],
    'Population': [276, 260, 269, 466],
    'Price': [120, 83, 80, 97],
    'Age': [42, 65, 59, 55],
    'US': ['Yes', 'Yes', 'Yes', 'Yes']
})

# Encode categorical variables: ShelveLoc, Urban, and US
data = pd.get_dummies(data, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

# Separate the features (X) and the target variable (y)
X = data.drop('Sales', axis=1)  # Features
y = data['Sales']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)

# Fit the regressor to the training data
regressor.fit(X_train, y_train)

# Predict using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the regressor using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Print the result
print(f'Mean Absolute Error (MAE): {mae}')
