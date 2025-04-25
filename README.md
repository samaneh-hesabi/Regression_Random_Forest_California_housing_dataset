<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Random Forest Regression for California Housing</div>

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
The project requires the following Python packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

# 5. Installation
You can install the dependencies using either pip or conda:

## Using pip:
```bash
pip install -r requirements.txt
```

## Using conda:
```bash
conda env create -f environment.yml
conda activate california_housing
```

# 6. Usage
Run the script:
```bash
python random_forest_regression.py
```

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

# 8. Code Structure
The code is organized into several functions:
- `load_data()`: Loads and prepares the dataset
- `create_dataframe()`: Creates a pandas DataFrame
- `train_model()`: Trains the Random Forest model
- `evaluate_model()`: Evaluates model performance
- `plot_results()`: Generates visualization plots
- `main()`: Orchestrates the entire workflow

# 9. Best Practices
- Type hints for better code readability and IDE support
- Comprehensive docstrings for all functions
- Modular code structure for better maintainability
- Consistent code formatting
- Reproducible environment setup
- Clear documentation

# 10. License
This project is open source and available for educational purposes.

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Toy Dataset for Random Forest Regression</div>

This project demonstrates a simple Random Forest regression model using a synthetic dataset. It's designed to help understand how Random Forest works before applying it to more complex datasets like the California housing dataset.

# 1. Files in the Project

## 1.1 `toy_dataset_regression.py`
This script:
- Generates a synthetic dataset with 2 features and a non-linear relationship
- Trains a Random Forest regressor on the data
- Evaluates the model using MSE and R-squared metrics
- Visualizes the results with two plots:
  - Actual vs Predicted values
  - Feature importance

# 2. How to Run
1. Make sure you have the required packages installed:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```
2. Run the script:
   ```bash
   python toy_dataset_regression.py
   ```

# 3. Dataset Description
The synthetic dataset is generated with the following characteristics:
- 1000 samples
- 2 input features (X1 and X2)
- Target variable (y) is created using a non-linear relationship:
  - y = 2*X1 + 3*X2 + 0.5*X1*X2 + noise
- Features are randomly generated between 0 and 10
- Gaussian noise is added to make the problem more realistic

# 4. Model Details
- Uses Random Forest Regressor with 100 trees
- 80% of data used for training, 20% for testing
- Model performance is evaluated using:
  - Mean Squared Error (MSE)
  - R-squared score

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Synthetic California Housing Dataset</div>

This project includes a synthetic dataset that mimics the California housing dataset, allowing for experimentation and learning without using the original dataset.

# 1. Files in the Project

## 1.1 `synthetic_california_housing.py`
This script:
- Generates a synthetic dataset with the same features as the California housing dataset
- Creates realistic relationships between features and house prices
- Trains a Random Forest regressor on the synthetic data
- Evaluates the model using MSE and R-squared metrics
- Visualizes the results with two plots:
  - Actual vs Predicted house values
  - Feature importance

# 2. Dataset Features
The synthetic dataset includes the same features as the original California housing dataset:
- MedInc: Median income in block group (in tens of thousands)
- HouseAge: Median house age in block group (in years)
- AveRooms: Average number of rooms per household
- AveBedrms: Average number of bedrooms per household
- Population: Block group population
- AveOccup: Average number of household members
- Latitude: Block group latitude
- Longitude: Block group longitude
- MedHouseVal: Median house value (target variable, in $100,000s)

# 3. How to Run
1. Make sure you have the required packages installed:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Run the script:
   ```bash
   python synthetic_california_housing.py
   ```

# 4. Data Generation Details
The synthetic data is generated with realistic distributions and relationships:
- Income follows a log-normal distribution
- House age is uniformly distributed between 1-52 years
- Room counts follow a normal distribution
- Location coordinates are within California's boundaries
- House prices are generated using a non-linear combination of features
- Realistic noise and interactions are added to make the data more realistic

# 5. Model Details
- Uses Random Forest Regressor with 100 trees
- 80% of data used for training, 20% for testing
- Model performance is evaluated using:
  - Mean Squared Error (MSE)
  - R-squared score
