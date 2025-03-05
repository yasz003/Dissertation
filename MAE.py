import os
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Paths to data
pinn_predictions_path = r'C:\Users\yacin\Downloads\PhysNet_Thermal_Models-main\predictions.csv'
nn_predictions_path = r'C:\Users\yacin\Downloads\PhysNet_Thermal_Models-main\predictions_nn.csv'
test_data_path = r'C:\Users\yacin\Downloads\PhysNet_Thermal_Models-main\PhysNet_Thermal_Models-main\data\TestY_data5.csv'

# Read the data
pinn_predictions = pd.read_csv(pinn_predictions_path)  # Load PiNN predictions
nn_predictions = pd.read_csv(nn_predictions_path)      # Load NN predictions
test_data = pd.read_csv(test_data_path)                # Load test data

# Extract relevant columns
test_temperature = test_data['Current_State']  # Current temperature state from test data
pinn_temperature = pinn_predictions['T_r_k+1']  # Future temperature predictions from PiNN
nn_temperature = nn_predictions['T_r_k+1']  # Future temperature predictions from NN

# Ensure all columns have the same length
min_length = min(len(test_temperature), len(pinn_temperature), len(nn_temperature))
test_temperature = test_temperature[:min_length]
pinn_temperature = pinn_temperature[:min_length]
nn_temperature = nn_temperature[:min_length]

# Calculate Mean Absolute Errors (MAE)
mae_pinn = mean_absolute_error(test_temperature, pinn_temperature)
mae_nn = mean_absolute_error(test_temperature, nn_temperature)

# Print the MAEs
print(f"MAE for PiNN Predictions: {mae_pinn}")
print(f"MAE for NN Predictions: {mae_nn}")
