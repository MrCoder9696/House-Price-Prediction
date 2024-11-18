import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# File path to your dataset
file_path = 'data.csv'

# Attempt to load the dataset
try:
    # Read the dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found in directory {os.getcwd()}.")
    print("Using a sample dataset instead for testing.")
    
    # Sample dataset
    data = {
        'bedrooms': [3, 4, 2],
        'bathrooms': [2, 3, 1],
        'sqft_living': [1500, 2000, 850],
        'sqft_lot': [4000, 5000, 3000],
        'floors': [1, 2, 1],
        'waterfront': [0, 1, 0],
        'view': [0, 1, 0],
        'condition': [3, 4, 2],
        'price': [300000, 450000, 200000]
    }
    df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Define feature matrix (X) and target variable (y)
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict the price of a new house
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]  # Input feature values
predicted_price = model.predict(new_data)
print("\nPredicted Price for new house:", predicted_price[0])
