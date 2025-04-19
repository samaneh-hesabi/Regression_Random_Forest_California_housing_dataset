<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Random Forest Regression Project</div>

# 1. Project Overview
This project implements a Random Forest Regression model using scikit-learn to predict California housing prices. The model is trained on the California Housing dataset and includes comprehensive data analysis, model evaluation, and visualization of results.

# 2. Project Structure
- `random_forest_regression.py`: Main Python script containing the implementation
- `regression_plot.png`: Visualization of actual vs predicted house values
- `feature_importance.png`: Visualization of feature importance
- `requirements.txt`: Python package dependencies
- `environment.yml`: Conda environment configuration

# 3. Features
- Implementation of Random Forest Regression using scikit-learn
- Uses the California Housing dataset with 8 features:
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude
- Model evaluation using MSE and R² metrics
- Feature importance analysis
- Visualization of predictions and feature importance
- Example predictions on new data

# 4. Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

# 5. Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the script:
```bash
python random_forest_regression.py
```

# 6. Output
The script will:
- Load and analyze the California Housing dataset
- Train a Random Forest Regression model with 100 estimators
- Display model performance metrics (MSE and R²)
- Show feature importance rankings
- Generate visualization plots:
  - Actual vs Predicted house values
  - Feature importance bar plot
- Make predictions on example new data

# 7. Results
The model performance can be evaluated through:
- Mean Squared Error (MSE)
- R-squared Score
- Feature importance rankings
- Visual comparison of actual vs predicted house values

# 8. License
This project is open source and available for educational purposes.
