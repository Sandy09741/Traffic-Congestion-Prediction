# Importing the packages for to use the certain code 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the historical traffic data
# Assuming a CSV file with columns: 'timestamp', 'traffic_volume', and 'external_factor'
data = pd.read_csv('traffic_data.csv')
# Your Python code for the traffic congestion prediction application goes here 
# Ensure to include data preprocessing, model training, evaluation, and prediction steps 

# Convert timestamp to datetime
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Extract features (hour of the day, day of the week) from the timestamp
data['hour'] = data['DateTime'].dt.hour
data['day_of_week'] = data['DateTime'].dt.dayofweek

# Assuming 'external_factor' is an external influence like weather, events, etc.

# Select features and target variable
features = ['hour', 'day_of_week', 'Junction']
target = 'Vehicles'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize predictions
plt.scatter(X_test['hour'], y_test, color='black', label='Actual')
plt.scatter(X_test['hour'], predictions, color='blue', label='Predicted')
plt.xlabel('Hour of the Day')
plt.ylabel('Traffic Volume or Vehicles')
plt.legend()
plt.show()

# Residual Plot
residuals = y_test - predictions
plt.scatter(X_test['hour'], residuals, color='red', label='Residuals')
plt.axhline(y=0, color='black', linestyle='--', label='Zero Residuals')
plt.xlabel('Hour of the Day')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# Feature Importance
coef = model.coef_
feature_names = X_train.columns
plt.barh(feature_names, coef)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
# Line Plot of Predictions vs Actual Values
plt.plot(X_test['hour'], y_test, color='black', label='Actual')
plt.plot(X_test['hour'], predictions, color='blue', label='Predicted')
plt.xlabel('Hour of the Day')
plt.ylabel('Traffic Volume or Vehicles')
plt.legend()
plt.show()
