
import pandas as pd
import matplotlib.pyplot as plt

# Paths to data
pinn_predictions_path = r'C:\Users\yacin\Downloads\PhysNet_Thermal_Models-main\predictions.csv'
nn_predictions_path = r'C:\Users\yacin\Downloads\PhysNet_Thermal_Models-main\predictions_nn.csv'
test_data_path = r'C:\Users\yacin\Downloads\PhysNet_Thermal_Models-main\PhysNet_Thermal_Models-main\data\TestY_data6.csv'

# Read the data
pinn_predictions = pd.read_csv(pinn_predictions_path)  # Load PiNN predictions
nn_predictions = pd.read_csv(nn_predictions_path)      # Load NN predictions
test_data = pd.read_csv(test_data_path)                # Load test data


time_steps = test_data['Time']  # Time column from the test data
test_temperature = test_data['Current_State']  # Current temperature state from test data
pinn_temperature = pinn_predictions['T_r_k+1']  # Future temperature predictions from PiNN
nn_temperature = nn_predictions['T_r_k+1']  # Future temperature predictions from NN


min_length = min(len(time_steps), len(test_temperature), len(pinn_temperature), len(nn_temperature))
time_steps = time_steps[:min_length]
test_temperature = test_temperature[:min_length]
pinn_temperature = pinn_temperature[:min_length]
nn_temperature = nn_temperature[:min_length]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(
    time_steps, 
    test_temperature, 
    label="Test Data", 
    linestyle="-", 
    linewidth=2
)
plt.plot(
    time_steps, 
    pinn_temperature, 
    label="PiNN Predictions", 
    linestyle="-", 
    linewidth=2
)
plt.plot(
    time_steps, 
    nn_temperature, 
    label="NN Predictions", 
    linestyle="--", 
    linewidth=2
)


plt.title("Comparison of Test Data, PiNN Predictions, and NN Predictions", fontsize=14)
plt.xlabel("Time(minutes)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)


plt.show()