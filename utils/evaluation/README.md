# Evaluation Module

This module provides functions for evaluating the performance of machine learning models.

## Functions

1. **`evaluate_model(y_true, y_pred)`**
   - Calculates various evaluation metrics to assess model performance.
   - **Parameters**:
     - `y_true` (array-like): The true target values.
     - `y_pred` (array-like): The predicted target values.
   - **Returns**:
     - A dictionary containing the following metrics:
       - `MSE`: Mean Squared Error.
       - `RMSE`: Root Mean Squared Error.
       - `MAE`: Mean Absolute Error.
       - `R²`: Coefficient of Determination.
       - `MAPE`: Mean Absolute Percentage Error.

## Usage
This function is used in the main Streamlit app to evaluate and display the performance of trained models.