import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['MEDV'] = y  # MEDV is the target variable (median house value)

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("\nFeature names:")
for i, name in enumerate(feature_names):
    print(f"{i+1}. {name}")

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
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Print feature importance
print("\nFeature Importance:")
for name, importance in zip(feature_names, rf_model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Value (in $100,000)')
plt.ylabel('Predicted House Value (in $100,000)')
plt.title('Actual vs Predicted House Values')
plt.tight_layout()
plt.savefig('regression_plot.png')
plt.close()

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_model.feature_importances_, y=feature_names)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Example of predicting new data
# Let's use the first two samples from the test set as examples
new_data = X_test[:2]
new_predictions = rf_model.predict(new_data)
print("\nPredictions for new data:")
for i, pred in enumerate(new_predictions):
    print(f"Sample {i+1}: ${pred*100000:.2f}")
