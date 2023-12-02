import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Sample car dataset (replace this with your own dataset)
data = pd.DataFrame({
    'Car_Name': ['Honda City', 'Maruti Swift', 'Hyundai Creta', 'Toyota Corolla', 'Ford EcoSport'],
    'Year': [2017, 2016, 2018, 2015, 2017],
    'Fuel Type': ['Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel'],
    'Transmission': ['Manual', 'Manual', 'Automatic', 'Manual', 'Manual'],
    'Owner': [0, 0, 0, 0, 0],
    'Is_Fast': [1, 0, 1, 0, 1]  # Binary target variable (1 for fast cars, 0 for others)
})

# Encode categorical variables: Fuel Type and Transmission
label_encoder = LabelEncoder()
data['Fuel Type'] = label_encoder.fit_transform(data['Fuel Type'])
data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

# Separate features (X) and target variable (y)
X = data[['Year', 'Fuel Type', 'Transmission', 'Owner']]
y = data['Is_Fast']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a Support Vector Classifier (SVC)
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# Evaluate the SVC model
accuracy = accuracy_score(y_test, y_pred)

# Suppress the "UndefinedMetricWarning" warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Calculate the classification report without warnings
report = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy}')
print("\nClassification Report:")
print(report)
