"""
Random Forest Regression implementation for California Housing dataset.
This script implements a Random Forest Regression model to predict California housing prices.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from typing import Tuple, Any
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load and prepare the California Housing dataset.
    
    Returns:
        Tuple containing:
        - X: Feature matrix
        - y: Target values
        - feature_names: List of feature names
        
    Raises:
        Exception: If there's an error loading the dataset
    """
    try:
        logging.info("Loading California Housing dataset...")
        housing = fetch_california_housing()
        logging.info("Dataset loaded successfully")
        return housing.data, housing.target, housing.feature_names
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def create_dataframe(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the dataset.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        
    Returns:
        DataFrame containing features and target
        
    Raises:
        ValueError: If input arrays have incompatible shapes
    """
    try:
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must match")
        df = pd.DataFrame(X, columns=feature_names)
        df['MEDV'] = y  # MEDV is the target variable (median house value)
        return df
    except Exception as e:
        logging.error(f"Error creating DataFrame: {str(e)}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        logging.info(f"Training Random Forest model with {n_estimators} estimators...")
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate the model performance.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Tuple containing MSE and R² scores
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model evaluation completed - MSE: {mse:.4f}, R²: {r2:.4f}")
        return mse, r2
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise

def plot_results(y_test: np.ndarray, y_pred: np.ndarray, 
                feature_names: list, feature_importances: np.ndarray) -> None:
    """
    Generate and save visualization plots.
    
    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        feature_names: List of feature names
        feature_importances: Feature importance scores
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        # Plot actual vs predicted values
        logging.info("Generating regression plot...")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual House Value (in $100,000)')
        plt.ylabel('Predicted House Value (in $100,000)')
        plt.title('Actual vs Predicted House Values')
        plt.tight_layout()
        plt.savefig('regression_plot.png')
        plt.close()
        logging.info("Regression plot saved successfully")

        # Plot feature importance
        logging.info("Generating feature importance plot...")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_names)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        logging.info("Feature importance plot saved successfully")
    except Exception as e:
        logging.error(f"Error generating plots: {str(e)}")
        raise

def get_feature_importances(model: RandomForestRegressor, feature_names: list) -> pd.DataFrame:
    """
    Get feature importances from the Random Forest model.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame containing feature names and their importance scores
    """
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        return importance_df
    except Exception as e:
        logging.error(f"Error getting feature importances: {str(e)}")
        raise

def predict_new_data(model: RandomForestRegressor, new_data: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Make predictions on new data using the trained model.
    
    Args:
        model: Trained Random Forest model
        new_data: New data points to make predictions on
        feature_names: List of feature names for reference
        
    Returns:
        Array of predicted values
        
    Raises:
        ValueError: If new_data shape doesn't match expected features
    """
    try:
        if new_data.shape[1] != len(feature_names):
            raise ValueError(f"New data must have {len(feature_names)} features")
        predictions = model.predict(new_data)
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise

def main() -> None:
    """Main function to run the Random Forest Regression analysis."""
    try:
        # Load data
        X, y, feature_names = load_data()
        df = create_dataframe(X, y, feature_names)
        
        # Display dataset information
        logging.info("\nDataset Information:")
        logging.info(f"Number of samples: {X.shape[0]}")
        logging.info(f"Number of features: {X.shape[1]}")
        logging.info("\nFeature names:")
        for i, name in enumerate(feature_names):
            logging.info(f"{i+1}. {name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        mse, r2 = evaluate_model(model, X_test, y_test)
        logging.info(f"\nMean Squared Error: {mse:.4f}")
        logging.info(f"R-squared Score: {r2:.4f}")
        
        # Get and display feature importances
        importance_df = get_feature_importances(model, feature_names)
        logging.info("\nFeature Importances:")
        logging.info(importance_df.to_string(index=False))
        
        # Generate plots
        plot_results(y_test, model.predict(X_test), feature_names, model.feature_importances_)
        
        # Example predictions on new data
        logging.info("\nMaking predictions on new data:")
        # Create some example new data points
        new_data = np.array([
            [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23],  # Example 1
            [5.6431, 52.0, 7.984127, 1.123810, 240.0, 2.555556, 37.88, -122.23]   # Example 2
        ])
        
        # Make predictions
        new_predictions = predict_new_data(model, new_data, feature_names)
        
        # Display predictions
        logging.info("\nPredictions for new data points:")
        for i, (data, pred) in enumerate(zip(new_data, new_predictions)):
            logging.info(f"\nSample {i+1}:")
            for name, value in zip(feature_names, data):
                logging.info(f"{name}: {value:.4f}")
            logging.info(f"Predicted House Value: ${pred*100000:.2f}")
            
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
