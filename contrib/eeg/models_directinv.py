"""
Neural network models for direct inversion

LSTM model from 
Hecker, L., Rupprecht, R., Elst, L. T. van, & Kornmeier, J. (2022). 
Long-Short Term Memory Networks for Electric Source Imaging with Distributed Dipole Models (p. 2022.04.13.488148).
bioRxiv. https://doi.org/10.1101/2022.04.13.488148
"""

import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from contrib.eeg.models import CosineReshape


class HeckerLSTM(nn.Module): 
    def __init__(self, n_electrodes=61, hidden_size=85, n_sources=1284, bias=False) -> None:
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size = n_electrodes, hidden_size = hidden_size, num_layers = 2, dropout = 0.2, bidirectional = True)
        self.fc = torch.nn.Linear( hidden_size*2, n_sources, bias=bias )
        
    def forward(self, x): 
        out, _ = self.lstm( einops.rearrange(x, 'b c t -> t b c' ) ) # match the input dimension from LSTM layer in pytorch 
        out = F.relu(out)
        
        out = self.fc(  einops.rearrange(out, 't b c -> b t c' ) ) # pass input sequentially in fully connected layer
        
        return einops.rearrange(out, 'b t c -> b c t' ) #to get original data dimensions


class HeckerLSTMpl( pl.LightningModule ): 
    def __init__(self, n_electrodes=61, hidden_size=85, 
            n_sources=1284, bias=False, 
            optimizer = torch.optim.Adam, lr = 0.001,  criterion = CosineReshape(), 
            mc_dropout_rate=0) -> None:
        super().__init__()
        self.mc_dropout_rate = mc_dropout_rate
        
        self.model = HeckerLSTM(n_electrodes=n_electrodes, hidden_size=hidden_size, n_sources=n_sources ,bias=bias)

        self.criterion = criterion
        # self.criterion = CosineReshape()
        self.optimizer = optimizer
        self.lr = lr


    def forward(self, x): 
        return self.model(x)
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

        # compute loss
        loss = self.criterion(src_hat, src)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss = self.criterion(src_hat, src)

        self.log("validation_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def predict_step(self, batch) :
        self.model.dropout.train()
        preds = self.model(batch)
        return preds


"""
feb. 2023 - network from the article: 
Sun, R., Sohrabpour, A., Worrell, G. A., & He, B. (2022). 
Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics.
Proceedings of the National Academy of Sciences, 119(31), e2201128119. https://doi.org/10.1073/pnas.2201128119 

code from the github repo https://github.com/bfinl/DeepSIF/

"""
class MLPSpatialFilter(nn.Module):
    def __init__(self, num_sensor, num_hidden, activation):
        super(MLPSpatialFilter, self).__init__()
        self.fc11 = nn.Linear(num_sensor, num_sensor)
        self.fc12 = nn.Linear(num_sensor, num_sensor)
        self.fc21 = nn.Linear(num_sensor, num_hidden)
        self.fc22 = nn.Linear(num_hidden, num_hidden)
        self.fc23 = nn.Linear(num_sensor, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        x = self.activation(self.fc12(self.activation(self.fc11(x))) + x)
        x = self.activation(self.fc22(self.activation(self.fc21(x))) + self.fc23(x))
        out['value'] = self.value(x)
        out['value_activation'] = self.activation(out['value'])
        return out

class TemporalFilter(nn.Module):
    def __init__(self, input_size, num_source, num_layer, activation):
        super(TemporalFilter, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size, num_source, batch_first=True, num_layers=num_layer))
        self.num_layer = num_layer
        self.input_size = input_size
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        # c0/h0 : num_layer, T, num_out
        for l in self.rnns:
            l.flatten_parameters()
            x, _ = l(x)

        out['rnn'] = x  # seq_len, batch, num_directions * hidden_size
        return out

class TemporalInverseNet(nn.Module):
    def __init__(self, num_sensor=64, num_source=994, rnn_layer=3,
                 spatial_model=MLPSpatialFilter, temporal_model=TemporalFilter,
                 spatial_output='value_activation', temporal_output='rnn',
                 spatial_activation='ReLU', temporal_activation='ReLU', temporal_input_size=500):
        super(TemporalInverseNet, self).__init__()
        self.attribute_list = [num_sensor, num_source, rnn_layer,
                               spatial_model, temporal_model, spatial_output, temporal_output,
                               spatial_activation, temporal_activation, temporal_input_size]
        self.spatial_output = spatial_output
        self.temporal_output = temporal_output
        # Spatial filtering
        self.spatial = spatial_model(num_sensor, temporal_input_size, spatial_activation)
        # Temporal filtering
        self.temporal = temporal_model(temporal_input_size, num_source, rnn_layer, temporal_activation)

    def forward(self, x):
        out = dict()
        out['fc2'] = self.spatial(x)[self.spatial_output]
        x = out['fc2']
        out['last'] = self.temporal(x)[self.temporal_output]
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class DeepSIFpl(pl.LightningModule): 
    def __init__(self, 
        num_sensor = 64, num_source = 994, temporal_input_size = 500, 
        optimizer = torch.optim.Adam, 
        lr = 0.001, 
        criterion = torch.nn.MSELoss(), rnn_layer=3) -> None:
        super().__init__()

        self.deep_sif_params = dict( 
            num_sensor = num_sensor, 
            num_source =num_source, 
            temporal_input_size = temporal_input_size,
            rnn_layer = rnn_layer,
            spatial_output='value_activation', temporal_output='rnn',
            spatial_activation='ReLU', temporal_activation='ReLU'
        )
        
        self.optimizer = optimizer 
        self.lr = lr 
        self.criterion = criterion 

        self.model = TemporalInverseNet( 
            **self.deep_sif_params
        )


    def forward(self, x): 
        out = self.model(torch.permute(x, (0,2,1)))["last"]
        return torch.permute( out, (0,2,1))
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)

        loss = self.criterion(src_hat, src)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg = eeg.float()
        src = src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss = self.criterion(src_hat, src)

        self.log("validation_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

###-----------------------CNN1D------------------------------###
class Wide1dCNN(nn.Module):
  """
  First CNN :
  2 convolutional layers (1d) + non linear activation function btwn them 
  (= minimum to be able to approximate a non linear function)

  activation = Relu

  inputs: 
    channels : list of values of input and output channels for the different layers (ex: n_electrodes, 4*n_electrodes, n_sources)
     if there are N layers, channels should be N+1 long (2 layers: 2+1=3 values indeed)
    kernel_size: size of the kernel for convolutions
    bias : bias for convolutions

  """ 
  def __init__(self, channels, kernel_size = 3, bias = False, sum_xai=False):
        super().__init__()
        self.sum_xai = sum_xai
        self.conv1  = nn.Conv1d( 
          in_channels   = channels[0], 
          out_channels  = channels[1], 
          kernel_size   = kernel_size, 
          dilation      = 1, 
          padding       = 'same', 
          bias          = bias)
        self.fc     = nn.Linear(
          in_features   = channels[1], 
          out_features  = channels[2], 
          bias = bias )

  def forward(self, x):
    x = self.conv1(x)
    output = self.fc( F.relu( torch.permute(x, (0,2,1)) ) )
    #output = F.relu(x)
    if self.sum_xai : 
      return torch.permute( output, (0,2,1) ).sum(dim=2)
    else : 
      return torch.permute( output, (0,2,1) )
    
## Lighting module
class CNN1Dpl(pl.LightningModule ): 
    def __init__(self, channels, kernel_size = 5, bias=False,  optimizer = torch.optim.Adam, lr = 0.001, 
          criterion = CosineReshape(), sum_xai=False) -> None:

        super().__init__()
        self.model = Wide1dCNN(channels, kernel_size, bias, sum_xai=sum_xai)
        self.criterion = criterion
        # self.criterion=CosineReshape()
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, x): 
        return self.model(x)
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss_train = self.criterion(src_hat, src)
        self.log("train_loss", loss_train, prog_bar=True, on_step=False, on_epoch=True)    

        return loss_train
      
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(eeg)

        # compute loss
        loss_val = self.criterion(src_hat, src)
        self.log("validation_loss", loss_val, prog_bar=True, on_step=False, on_epoch=True)

        return loss_val