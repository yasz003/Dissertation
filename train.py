
import warnings
import logging
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from utils.support_functions import (
    transform_temp,
    transform_action,
    transform_outside_temp,
    inverse_transform_temp,
    inverse_transform_action,
    inverse_transform_outside_temp
)

from utils.global_state_variables import MAX_TIME

from PhysNet import PhysNet as PhysNet
from PhysRegMLP import PhysRegMLP as PhysRegMLP

# To stop PyTorch Lightning from giving GPU Availability and warning messages
warnings.filterwarnings('ignore')
logging.getLogger('lightning').setLevel(0)


# -------------------------------------------------------------------------------------------------------------------- #

def prepare_data(data):
    x_inputs = data['x_agg_k']
    y_labels = data['label_k']

    # Check if horizon_size of inputs and outputs is the same
    if x_inputs.shape[0] != y_labels.shape[0]:
        raise ValueError("Size of inputs and labels should be the same")

    state_depth = (x_inputs.shape[1] - 4) // 2
    # Scaling
    x_inputs[:, 0] = x_inputs[:, 0] / MAX_TIME  # Time
    x_inputs[:, 1] = transform_temp(x_inputs[:, 1])  # Current Temp
    x_inputs[:, 2:(2 + state_depth)] = transform_temp(x_inputs[:, 2:(2 + state_depth)])  # Previous temp states
    x_inputs[:, (2 + state_depth):(2 + 2 * state_depth)] = transform_action(
        x_inputs[:, (2 + state_depth):(2 + 2 * state_depth)])  # Previous Actions
    x_inputs[:, -2] = transform_action(x_inputs[:, -2])  # Current Action
    x_inputs[:, -1] = transform_outside_temp(x_inputs[:, -1])  # Outside Temp

    y_labels[:, 0] = transform_temp(y_labels[:, 0])
    y_labels[:, 1] = transform_action(y_labels[:, 1])

    return {"x_agg_k": np.array(x_inputs[0:-1]),
            "label_k": np.array(y_labels[0:-1]),
            "x_agg_k1": np.array(x_inputs[1:]),
            "label_k1": np.array(y_labels[1:])}


def preprocess_test_data(test_data, depth):
   
    test_data[:, 0] = test_data[:, 0] / MAX_TIME  # Time
    test_data[:, 1] = transform_temp(test_data[:, 1])  # Current Temp
    test_data[:, 2:(2 + depth)] = transform_temp(test_data[:, 2:(2 + depth)])  # Previous temp states
    test_data[:, (2 + depth):(2 + 2 * depth)] = transform_action(
        test_data[:, (2 + depth):(2 + 2 * depth)])  # Previous Actions
    test_data[:, -2] = transform_action(test_data[:, -2])  # Current Action
    test_data[:, -1] = transform_outside_temp(test_data[:, -1])  # Outside Temp
    return test_data


def predict_and_save(model_instance, test_data, save_path):
  
    predictions = []
    for i in range(len(test_data)):
        test_sample = torch.tensor(test_data[i], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        model_output = model_instance(test_sample)
        
        # Unpack the tuple if necessary
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # Assume the first element contains the predictions
        
        # Ensure the output is squeezed to remove extra dimensions
        model_output = model_output.squeeze().detach().numpy()
        predictions.append(model_output)
    
    # Stack predictions to form a 2D array
    predictions = np.vstack(predictions)  # Shape should now be (N, 2)
    
    # Apply inverse transformations to predictions
    predictions[:, 0] = inverse_transform_temp(predictions[:, 0])  # For T_r_k+1
    predictions[:, 1] = inverse_transform_action(predictions[:, 1])  # For u_phys_k
    
    # Debugging: Print predictions
    print("Raw Predictions (First 5 Rows):", predictions[:5])
    
    # Convert to DataFrame and save
    predictions_df = pd.DataFrame(predictions, columns=["T_r_k+1", "u_phys_k"])
    predictions_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


# -------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    # Simulation Params
    depth = 8
    lambda_2 = 0.5
    model_type = 'PhysNet'  # PhysNet or PhysRegMLP
    model_seed = 1

    if model_type == 'PhysRegMLP':
        model = PhysRegMLP
        network_param = {
            'lr': 0.001,
            'batch_size': 2048,
            'lambda_value': lambda_2,
            'mdp_network': {'input_size': 4 + 2 * (depth + 0),
                            'fc': [64] * 2,
                            'output_size': 3,
                            'activation': 'tanh',
                            'dropout_rate': 0.01}
        }
    elif model_type == 'PhysNet':
        model = PhysNet
        network_param = {
            'lr': 0.001,
            'batch_size': 2048,
            'lambda_value': lambda_2,
            'encoding_network': {'input_size': 2 * (depth + 0),
                                 'fc': [24] * 1,
                                 'output_size': 1,
                                 'activation': 'tanh',
                                 'dropout_rate': 0.01},
            'mdp_network': {'input_size': 5,
                            'fc': [128] * 1,
                            'output_size': 2,
                            'activation': 'tanh',
                            'dropout_rate': 0.05}
        }
    else:
        raise TypeError("Incorrect Model Type")

    # Load Training data
    training_data_df = pd.read_csv('C:/Users/yacin/Downloads/PhysNet_Thermal_Models-main/PhysNet_Thermal_Models-main/data/Yacine_experiment4.csv')
    training_data_main = training_data_df.to_numpy()

    training_data_index_selection = tuple(
        [0, 1, *list(np.arange(4, 4 + depth)), *list(np.arange(4 + 24, 4 + 24 + depth)), 2,
         -1])
    input_data_dict = {'x_agg_k': training_data_main[:, training_data_index_selection],
                       'label_k': training_data_main[:, (3, -2)]}

    training_data_dict = prepare_data(input_data_dict)

    # Load Test data
    test_data_df = pd.read_csv('C:/Users/yacin/Downloads/PhysNet_Thermal_Models-main/PhysNet_Thermal_Models-main/data/TestY_data6.csv')
    test_data_main = test_data_df.to_numpy()

    test_data_index_selection = tuple(
        [0, 1, *list(np.arange(4, 4 + depth)), *list(np.arange(4 + 24, 4 + 24 + depth)), 2,
         -1])
    test_data = test_data_main[:, test_data_index_selection]

    # Preprocess Test Data
    test_data = preprocess_test_data(test_data, depth)

    # Seed and Initialize model
    torch.manual_seed(seed=model_seed)
    model_instance = model(network_param)

    # Add data to model
    model_instance.add_training_data(training_data_dict)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=75,
        min_epochs=1,
        accelerator='cpu',
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(model_instance)

    # Predict and save results
    output_path = 'C:/Users/yacin/Downloads/PhysNet_Thermal_Models-main/predictions.csv'
    predict_and_save(model_instance, test_data, output_path)
