## Functions to use graphs
from contrib.eeg.data import EsiDatamodule, EsiDatasetAE
import torch
## create the dataset
from contrib.eeg import utils_eeg
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch.nn as nn

def compute_neighbs(fwd): 
    return utils_eeg.get_neighbors(
        [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
        [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
    )

def get_edge_index(fwd): 
    """
    Compute the edge_index matrix from the mne-python Forward object of the head model.
    edge_index: dim 2,n_edges -> (i,j) where nodes i and j in the mesh shares an edge (are neighbors)
    = sparse representation of the adjacency matrix
    """
    neighbors = compute_neighbs(fwd)
    ## compute the adjacency and edge index matrices: 
    n_sources = neighbors.shape[0]
    A = np.zeros((n_sources, n_sources)) #adjency matrix
    for i in range(n_sources):
        for n in neighbors[i]: 
            if n >= 0:
                A[i,n] = 1
        A[i,i]=1

    n_edges = int( (A.sum() - n_sources)//2 )
    print(f"{n_edges=}")

    edge_index = torch.zeros( (2, n_edges) )
    k = 0
    for i in range(n_sources): 
        for j in range(i): 
            if A[i,j] == 1: 
                edge_index[0, k] = i
                edge_index[1, k] = j
                k += 1 
    return edge_index.long()

from torch_geometric.data import Data
class EsiDatasetGraph(EsiDatasetAE): 
    def __init__(self, datafolder, config_file, simu_name, subject_name, 
                 source_sampling, electrode_montage, to_load, snr_db, 
                 noise_type={ "white": 1 }, scaler_type="linear", orientation="constrained", 
                 neighbors = None, replace_root=False, load_lf=True, 
                 subset_file=None, add_noise=True):
        super().__init__(datafolder, config_file, simu_name, subject_name, source_sampling, 
                         electrode_montage, to_load, snr_db, noise_type, scaler_type, orientation, 
                         replace_root, load_lf, subset_file, add_noise)
        ## compute edge_index: 
        n_sources = neighbors.shape[0]
        A = np.zeros((n_sources, n_sources)) #adjency matrix
        for i in range(n_sources):
            for n in neighbors[i]: 
                if n >= 0:
                    A[i,n] = 1
            A[i,i]=1
        n_edges = int( (A.sum() - n_sources)//2 )
        self.edge_idx = torch.zeros((2,n_edges))
        k = 0
        for i in range(n_sources): 
            for j in range(0, i): 
                if A[i,j] == 1: 
                    self.edge_idx[0, k] = i
                    self.edge_idx[1, k] = j
                    k += 1
        self.edge_idx = self.edge_idx.long()
        
    def __getitem__(self, index):
        src, _ = super().__getitem__(index)
        if self.add_noise:
            # add noise to data - random snr in a range
            noise = torch.randn_like(src) 
            snr_db_range = np.arange(-15, 0, 1)
            snr_db = np.random.choice(snr_db_range)
            snr = 10**(snr_db/10)

            alpha_snr = (1/np.sqrt(snr))*(src.norm() / noise.norm())
            data_noisy = src + alpha_snr*noise
            return Data( x=data_noisy, edge_index=self.edge_idx), src
        else : 
            return Data(x=src, edge_index=self.edge_idx), src

class EsiDatamoduleGraph(EsiDatamodule):
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2, config_file=None, subset_name=None, neighbors=None):
        super().__init__(dataset_kw, dl_kw, per_valid, config_file, subset_name)
        self.neighbors = neighbors

    def setup(self, stage) : 
        if stage == "test": 
            self.test_ds = EsiDatasetGraph(
                **self.dataset_kw, neighbors=self.neighbors
            )
        else : 
            ds_dataset = EsiDatasetGraph(
                **self.dataset_kw, neighbors=self.neighbors
            )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            )

import pytorch_lightning as pl
from torch_geometric.nn import GraphUNet
from contrib.eeg.models import CosineReshape
## graph autoencoder prior: 
class GAEPrior(nn.Module): 
    def __init__(self,gae_params, edge_index, cost_fn=CosineReshape(), device="cuda"):
        super().__init__()
    
        self.model = GraphUNet( **gae_params).to(device=device)
        self.cost_fn = cost_fn
        self.edge_index = edge_index.to(device=next(self.model.parameters()).device)
        print(self.edge_index.device)
    def forward_ae(self, x): 
        # x: shape batch_size x n_sources x time
        # put in torch geomtric batch data format
        # 
        edge_index = self.edge_index.to(device=x.device)
        data_list = [Data(x=xi, edge_index=edge_index) for xi in x]
        gbatch = Batch.from_data_list(data_list)

        return self.model( gbatch.x, gbatch.edge_index ).view(x.shape)
    
    def forward(self, x): 
        # print(self.cost_fn( x, self.forward_ae(x) ))
        return self.cost_fn( x, self.forward_ae(x) )

class GAE_pl(pl.LightningModule ): 
    ## Based on the graph Unet model from Gao et al., implemented in torch geometric
    def __init__(self, gae_params_dict, 
                    optimizer = torch.optim.Adam, lr = 0.001, 
                    criterion = torch.nn.MSELoss(), **kwargs) -> None:
        super().__init__()
        
        self.model = GraphUNet(**gae_params_dict)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.kwargs = kwargs

    def forward(self, x):
        # x = Data object from pytorch geomtric
        return self.model(x.x, x.edge_index)
        
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        g_src, src = batch
        # g_src.x = g_src.x.float()
        src = src.float()
        src_hat = self.forward(g_src).float()

        # compute loss
        loss_train = self.criterion(src_hat, src)
        self.log("train_loss", loss_train, prog_bar=True, on_step=False, on_epoch=True)
        return loss_train
    
    def validation_step(self, batch, batch_idx):
        g_src, src = batch
        # g_src.x = g_src.x.float()
        src = src.float()
        src_hat = self.forward(g_src).float()
        # compute loss
        loss_val = self.criterion(src_hat, src)
        self.log("val_loss", loss_val, prog_bar=True, on_step=False, on_epoch=True)
       
        return loss_val


######### 
from torch_geometric.nn import GraphUNet
from torch_geometric.data import Batch
from torch_geometric.data import Data

class GraphGradModel(nn.Module):
    ## GAE for gradient descent model
    def __init__(self, edge_index, dim_in, dim_hidden=32, depth=3, pool_ratios = 0.5):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.edge_index = edge_index
        self.gae = GraphUNet( 
            in_channels = dim_in, 
            hidden_channels = dim_hidden, 
            out_channels = dim_in, 
            depth=depth, 
            pool_ratios = pool_ratios
        )
    def reset_state(self, inp):
        self._grad_norm = None


    def forward(self, x):
        ## normalisation de l'input par le gradient du la première itération
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
        # move to device
        edge_index = self.edge_index.to(device=x.device)
        # data_list = [
        #     Data( x=x[i,:,:], edge_index=self.edge_index.to(device=x.device))
        #     for i in range(x.size(0))]
        data_list = [Data(x=xi, edge_index=edge_index) for xi in x]
        gbatch = Batch.from_data_list(data_list)

        return self.gae( gbatch.x, gbatch.edge_index ).view(x.shape)

########### graph conv lstm ###################
## modifications from the code of the pytorch geometric temporal library ##
from typing import Tuple

import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric_temporal import GConvLSTM 
from torch_geometric.nn.conv import GCNConv

class GConvLSTMGradModel(nn.Module):
    ## modification of the gLSTM model to use an other type of graph convolution and use it as gradient model
    # original source code from torch geometric, with modifications.
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim_hidden: int,
        edge_index:  torch.LongTensor,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvLSTMGradModel, self).__init__()

        self.edge_index=edge_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_hidden = dim_hidden
        self.K = 1
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()
        self.relu = nn.ReLU()

        self._state = [] ## added based on the convLSTM grad model

        self.conv_out = GCNConv(in_channels=dim_hidden, out_channels=out_channels, add_self_loops=True)
    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.dim_hidden).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.dim_hidden).to(X.device)
        return C
    def reset_state(self, inp): 
        ## added based on the convLSTM grad model
        edge_index = self.edge_index.to(device=inp.device)
        data_list = [Data(x=xi, edge_index=edge_index) for xi in inp]
        gbatch = Batch.from_data_list(data_list)

        self._grad_norm = None
        self._state = [ 
            self._set_hidden_state(gbatch.x, None), 
            self._set_cell_state(gbatch.x, None)
        ]
    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)
    
    ### replace chebconv with GCNconv
    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )
        self.conv_h_i = GCNConv(
            in_channels=self.dim_hidden,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )
        self.w_c_i = Parameter(torch.Tensor(1, self.dim_hidden))
        self.b_i = Parameter(torch.Tensor(1, self.dim_hidden))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )
        self.conv_h_f = GCNConv(
            in_channels=self.dim_hidden,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.dim_hidden))
        self.b_f = Parameter(torch.Tensor(1, self.dim_hidden))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )
        self.conv_h_c = GCNConv(
            in_channels=self.dim_hidden,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )
        self.b_c = Parameter(torch.Tensor(1, self.dim_hidden))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )
        self.conv_h_o = GCNConv(
            in_channels=self.dim_hidden,
            out_channels=self.dim_hidden,
            bias=self.bias,
            improved=True, 
            add_self_loops=True
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.dim_hidden))
        self.b_o = Parameter(torch.Tensor(1, self.dim_hidden,))
    
    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index)#, edge_weight) #, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index)#, edge_weight) #, lambda_max=lambda_max)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index)#, edge_weight) #, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index)#, edge_weight) #, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index) #, edge_weight) #, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index) #, edge_weight) #, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index) #, edge_weight) #, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index) #, edge_weight) #, lambda_max=lambda_max)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O
    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H
    
    def forward(
        self,
        X: torch.FloatTensor,
        # edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        # H: torch.FloatTensor = None,
        # C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]: #, torch.FloatTensor]:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        ## modified based on the convLSTM grad model
        #H = self._set_hidden_state(X, H)
        #C = self._set_cell_state(X, C)
        if self._grad_norm is None:
            self._grad_norm = (X**2).mean().sqrt()
        X =  X / self._grad_norm
        H, C = self._state
        ### 

        edge_index = self.edge_index.to(device=X.device)
        data_list = [Data(x=xi, edge_index=edge_index) for xi in X]
        gbatch = Batch.from_data_list(data_list)

        I = self._calculate_input_gate(gbatch.x, gbatch.edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(gbatch.x, gbatch.edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(gbatch.x, gbatch.edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(gbatch.x, gbatch.edge_index, edge_weight, H, C, lambda_max)
        
        
        H = self._calculate_hidden_state(O, C)
        ## modified on the convLSTM grad model
        self._state = H, C
        ## output convolution
        out = self.conv_out(H, gbatch.edge_index)
        return out.view( X.shape )
