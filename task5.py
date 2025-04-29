import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Create a synthetic dataset
np.random.seed(42)
data_size = 200

X1 = np.random.rand(data_size) * 10  # Feature 1
X2 = np.random.rand(data_size) * 5   # Feature 2
Y = 2.5 * X1 + 1.2 * X2 + np.random.randn(data_size) * 2  # Linear relationship with noise

df = pd.DataFrame({"Feature1": X1, "Feature2": X2, "Target": Y})
df.to_csv("linear_regression_dataset.csv", index=False)  # Save dataset to CSV

print(df.head())  # Display sample dataset
# Using only Feature1 for Simple Linear Regression
X = df[["Feature1"]]
y = df["Target"]

# Splitting data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Simple Regression): {mse:.2f}")

# Plot results
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Predicted Regression Line")
plt.xlabel("Feature1")
plt.ylabel("Target")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
# Using Feature1 & Feature2 for Multiple Linear Regression
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# Splitting data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Multiple Regression): {mse:.2f}")

# Model Coefficients
print("Regression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)