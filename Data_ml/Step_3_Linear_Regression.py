# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import joblib

# Load the processed dataset
data = pd.read_csv("Step_3_Processed_Dataset.csv")

# Drop unnecessary columns to simplify the model
data = data.drop(['kitchen', 'Population', 'INS_Code'], axis=1)
print(data.columns.tolist())
print(data.shape)

# Function to remove outliers using the Interquartile Range (IQR) method
def remove_outliers_iqr(df, column_name):

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Remove outliers from the 'price' column
data = remove_outliers_iqr(data, 'price')

# Remove outliers from the 'livingArea' column
data = remove_outliers_iqr(data, 'livingArea')

# Define features (X) and target (y)
X = data.drop(columns=['price'])
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model and print training R² score
model = LinearRegression()
print(f"Model: {model}")
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'Step_4_Linear_Regression_Model.pkl')
joblib.dump(scaler, 'Step_4_Scaler.pkl')

# Predictions on training and test sets
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# MAE
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

# RMSE (Root Mean Squared Error)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# R² Score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# MAPE (Mean Absolute Percentage Error)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

# sMAPE (Symmetric Mean Absolute Percentage Error)
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

smape_train = smape(y_train, y_train_pred)
smape_test = smape(y_test, y_test_pred)

# Print the results
print(f"MAE (Training): {mae_train:.2f}, MAE (Test): {mae_test:.2f}")
print(f"RMSE (Training): {rmse_train:.2f}, RMSE (Test): {rmse_test:.2f}")
print(f"R² (Training): {r2_train:.2f}, R² (Test): {r2_test:.2f}")
print(f"MAPE (Training): {mape_train:.2f}%, MAPE (Test): {mape_test:.2f}%")
print(f"sMAPE (Training): {smape_train:.2f}%, sMAPE (Test): {smape_test:.2f}%")

# Perform K-Fold Cross-Validation
k = 5
scores = cross_val_score(model, X, y, cv=k, scoring='r2')

# Print the results
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Score: {np.mean(scores)}")

# Print Number of features used in the model
print(f"Number of features used in the model: {X_train_scaled.shape[1]}")

# Calculate feature importance as percentages
feature_importance = np.abs(model.coef_)
total_importance = np.sum(feature_importance)
percentage_importance = (feature_importance / total_importance) * 100

# Display feature importance in percentages
print("Feature Importance (%):")
for feature, importance in zip(X.columns, percentage_importance):
    print(f"{feature}: {importance:.2f}%")

# # Print model type and parameters
# print("Model Type:", type(model).__name__)
# print("Model Parameters:", model.get_params())
