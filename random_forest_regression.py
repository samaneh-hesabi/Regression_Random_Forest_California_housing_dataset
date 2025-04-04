import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate example data
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 2 * X[:, 0] + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(100)  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Print feature importance
print("\nFeature Importance:")
for i, importance in enumerate(rf_model.feature_importances_):
    print(f"Feature {i+1}: {importance:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('regression_plot.png')
plt.close()

# Example of predicting new data
new_data = np.array([[0.1, 0.2, 0.3],  # Example 1
                     [0.4, 0.5, 0.6]])  # Example 2

new_predictions = rf_model.predict(new_data)
print("\nPredictions for new data:")
for i, pred in enumerate(new_predictions):
    print(f"Sample {i+1}: {pred:.4f}")
