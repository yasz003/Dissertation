

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utils.support_functions import fc_module


class SimplifiedPhysNet(pl.LightningModule):
    def __init__(self, parameter_dict=None):
        super().__init__()

        if parameter_dict is None:
            parameter_dict = {
                'lr': 0.005, 'batch_size': 128,
                'encoding_network': {'input_size': 2,  # [x_t-1, x_t-2]
                                     'fc': [16, ],
                                     'output_size': 1,  # [T_m]
                                     'activation': 'tanh',
                                     'dropout_rate': 0.0},
                'mdp_network': {'input_size': 5,  # [Time, T_r, T_m, u_k, T_a_k]
                                'fc': [16, ],
                                'output_size': 2,  # [T_r_k+1, u_k]
                                'activation': 'tanh',
                                'dropout_rate': 0.0}
            }

        self.parameter_dict = parameter_dict
        self.encoding_network_params = parameter_dict['encoding_network']
        self.mdp_network_params = parameter_dict['mdp_network']

        self.encoding_network = nn.Sequential(*self.make_network(network_params=self.encoding_network_params))
        self.mdp_network = nn.Sequential(*self.make_network(network_params=self.mdp_network_params))

        # Model Parameters
        self.lr = parameter_dict['lr']
        self.batch_size = parameter_dict['batch_size']

        self.training_loss = {'Prediction Loss': [], 'Total Loss': []}

    @staticmethod
    def make_network(network_params):
        
        if len(network_params['fc']) == 0:
            network = [fc_module([network_params['input_size'], network_params['output_size']],
                                 activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
        else:
            network = [fc_module([network_params['input_size'], network_params['fc'][0]],
                                 activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
            for l_i in range(len(network_params['fc'][:-1])):
                network += [fc_module([network_params['fc'][l_i], network_params['fc'][l_i + 1]],
                                      activation=network_params['activation'],
                                      dropout_rate=network_params['dropout_rate'])]
            network += [fc_module([network_params['fc'][-1], network_params['output_size']],
                                  activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
        return network

    def forward(self, x1):
        
        x_state = x1[:, 2:-2]  # Extract [previous_states]
        x_T = x1[:, (0, 1)]  # Extract [Time, T_r]
        u_T_a = x1[:, (-2, -1)]  # Extract [u_k, T_a_k]

        # Encoded state
        x_M_k = self.encoding_network(x_state)

        # Predict next state
        x = torch.cat([x_T, x_M_k, u_T_a], dim=1)
        x_o_k1 = self.mdp_network(x)
        return x_o_k1, x_M_k

    @torch.no_grad()
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        o_k_1, _ = self.forward(x)
        return o_k_1.data.numpy()

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5),
                        'monitor': 'loss'}
        return [optimiser], [lr_scheduler]

    def training_step(self, batch, batch_idx):
      
        x_agg_k, o_k1, x_agg_k1, o_k2 = batch
        x_o_k1, _ = self.forward(x_agg_k)
        x_o_k2, _ = self.forward(x_agg_k1)

        # Compute loss (Mean Squared Error)
        prediction_loss = F.mse_loss(x_o_k1[:, 0], o_k1[:, 0]) + F.mse_loss(x_o_k2[:, 0], o_k2[:, 0])
        self.log('loss', prediction_loss)
        self.training_loss['Total Loss'].append(prediction_loss.data.numpy())
        return {'loss': prediction_loss}

    def add_training_data(self, main_data_dict):
        self.x_agg_k_data = main_data_dict['x_agg_k']
        self.label_k_data = main_data_dict['label_k']
        self.x_agg_k1_data = main_data_dict['x_agg_k1']
        self.label_k1_data = main_data_dict['label_k1']

    def train_dataloader(self):
        training_set = TensorDataset(torch.tensor(self.x_agg_k_data, dtype=torch.float32),
                                     torch.tensor(self.label_k_data, dtype=torch.float32),
                                     torch.tensor(self.x_agg_k1_data, dtype=torch.float32),
                                     torch.tensor(self.label_k1_data, dtype=torch.float32))
        training_data_loader = DataLoader(training_set, shuffle=False, batch_size=self.batch_size)
        return training_data_loader
