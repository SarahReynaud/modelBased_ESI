import pytorch_lightning as pl
import torch 
from torch import nn
import einops 
import torch.nn.functional as F
from src import models
import sys 
import numpy as np 

# define the different cost functions
class CosineReshape(nn.Module): 
    def __init__(self): 
        super().__init__()
    def forward(self, x, y): 
        return (1 - F.cosine_similarity(einops.rearrange(x, 'b c h -> b (c h)'), einops.rearrange(y, 'b c h -> b (c h)'))).mean()
class Cosine(nn.Module):
    def __init__(self): 
        super().__init__()
    def forward(self, x, y): 
        return (1 - F.cosine_similarity(x, y, dim=1) ).mean() # mean over batch of cosine similarity for each time point

# prior :
class ConvAEPrior(nn.Module): 
    """
    Convolutional autoencoder prior for source data.
    dim_in: Input dimension (number of input features).
    dim_hidden: Dimension of the hidden layer(s).
    dim_out: Output dimension (number of output features, typically equal to dim_in for reconstruction).
    bias: Whether to use bias terms in the convolutional layers.
    kernel_size: Size of the convolutional kernels (can be a single integer or a list of integers for different layers).
    cost_fn : cost function to use to compute the prior cost (x-phi(x))
    """
    def __init__(self, dim_in=1284, dim_hidden=128, dim_out=1284, bias=False, kernel_size=3,cost_fn=CosineReshape(), pretrained_model_path=None, fixed=False) -> None:
        super().__init__()
        if type(kernel_size) == int:
            kernel_size=[kernel_size]*3
        self.conv_in = nn.Conv1d(in_channels=dim_in, out_channels=dim_hidden, kernel_size=kernel_size[0], bias=bias, padding="same")
        self.conv_hidden = nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=kernel_size[1], bias=bias, padding="same")
        self.conv_out = nn.Conv1d(in_channels=dim_hidden, out_channels=dim_out, kernel_size=kernel_size[2], bias=bias, padding="same")
        self.cost_fn = cost_fn

        self.fixed = fixed        
        if pretrained_model_path is not None: # load a pretrained model if a path is provided
        
            loaded = torch.load(pretrained_model_path)
            new_state_dict = self.rename_keys(loaded['state_dict'], 'model.')
            self.load_state_dict(new_state_dict)
        
        if self.fixed : # freeze the model if fixed is True
            for param in self.parameters():
                param.requires_grad = False

    def forward_ae(self, x): 
        x = F.relu( self.conv_in(x) )
        x = F.relu( self.conv_hidden(x) )
        x = self.conv_out(x)
        return x
    
    def forward(self, x):
        # computes cost state x and reconstruction of x i.e prior cost
        return self.cost_fn(x,self.forward_ae(x))
    
    def forward_enc(self,x): 
        ## encode the input x
        x = F.relu( self.conv_in(x) )
        x = self.conv_hidden(x)
        return x
    
    def rename_keys(self,state_dict, prefix):
        """ 
        Rename keys in a state dict
        prefix: prefix to remove from the keys
        """
        new_state_dict = {}
        for key in state_dict.keys():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = state_dict[key]
            else :
                new_state_dict[key] = state_dict[key]
        return new_state_dict


## lightning module to train the autoencoder alone
class PriorPl(pl.LightningModule):
    def __init__(self, model, criterion=CosineReshape(), optimizer=torch.optim.Adam, lr=1e-3, pretrained_model_path=None, fixed=False, **kwargs) -> None:
        """
        model : nn Module initialized with the prior architecture
        criterion : loss function
        optimizer : optimizer
        lr : learning rate
        pretrained_model_path : path to a pretrained model
        fixed : whether the model should be frozen
        """
        super().__init__(**kwargs)
        self.model = model #ConvAEPrior(dim_in=dim_in, dim_hidden=dim_hidden, dim_out=dim_out, bias=bias, kernel_size=kernel_size,cost_fn=cost_fn)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr)
        self.fixed = fixed
        if pretrained_model_path is not None: # load a pretrained model if a path is provided
            loaded = torch.load(pretrained_model_path)
            self.load_state_dict(loaded['state_dict'])
        if self.fixed : # freeze the model if fixed is True
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        return self.model.forward_ae(x)

    def base_step(self, batch, batch_idx):
        x_in, x_tgt = batch
        out = self.model.forward_ae(x_in)
        loss = self.criterion(out, x_tgt)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.base_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx): 
        loss = self.base_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer


## obs cost : 
class EsiBaseObsCost(nn.Module):
    """
    Observation cost : D(Y, LX) where Y is the EEG data, L is the leadfield matrix and X the source data.
    - forward_obj: forward object from mne-python for which fwd['sol']['data'] contains the leadfield matrix.
    - device : device to use
    - cost_fn : cost function to use
    """
    def __init__(self, forward_obj, cost_fn = CosineReshape()) -> None:
        super().__init__()
        self.leadfield = torch.from_numpy( 
            forward_obj['sol']['data']).float()
        self.cost_fn = cost_fn
        self.fwd=forward_obj

    def forward(self, state, batch):
        """
        batch.input = EEG data
        state = estimated source data
        """
        return self.cost_fn(batch.input, torch.matmul(self.leadfield.to(device=state.device), state))

### grad mod (convLSTM)
class RearrangedConvLstmGradModel(models.ConvLstmGradModel):
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        super().reset_state(inp)

    def forward(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x
    
### solver
class EsiGradSolver(models.GradSolver) : 
    """
    wrapper around the GradSolver class, to use for ESI
    """
    def __init__(self, init_type = "zeros", fwd=None, mne_info=None, state_solver=False, reg_param={"prior": 1.0}, spatial_grad=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mne_info = mne_info
        self.init_type = init_type 
        self.fwd = fwd
        self.inv_op=None
        self.alpha_l = 1e3 # leadfield normalization if necessary
        self.reg = reg_param
        # ## check keys
        # if "spars" not in self.reg: 
        #     self.reg["spars"] = 0
        # if "grad_phi" not in self.reg:
        #     self.reg["grad_phi"] = 0
        # if "grad_state" not in self.reg: 
        #     self.reg["grad_phi"] = 0
        if "prior" not in self.reg: 
            self.reg["prior"] = 1.

        self.reg_mne = 1/9 # 1/snr^2
        self.state_solver = state_solver # which version of the grad_solver to use?
        
        ### compute gradient weight matrix
        self.L = spatial_grad

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        else :
            if self.init_type.upper() == "ZEROS": 
                state_0 = self.zeros_init(batch.tgt)
            elif self.init_type.upper() == "MNE":
                state_0 = self.mnep_init( batch.input )
            elif self.init_type.upper() == "NOISE": 
                state_0 = self.noise_init(batch.tgt)
            else : 
                sys.exit(f"{self.init_type=} unknown state init type")
            return state_0.detach().requires_grad_(True)
    
    def solver_step(self, state, batch, step):
        ## IF THE PRIOR IS A LSTM:
        # with torch.backends.cudnn.flags(enabled=False):

        prior_term = self.reg['prior'] * self.prior_cost(state)
        obs_term =  self.obs_cost(state, batch)
        # out_ae = self.prior_cost.forward_ae(state)       
        
        var_cost =  obs_term + prior_term
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0] # compute gradient of var cost w.r.t state
        
        ## Some tests on using both gradient and state as input to the gradient solver
        if not self.state_solver:
            gmod = self.grad_mod( grad ) 
        else: 
            gmod = self.grad_mod( grad, state ) #state ou bien state.detach??

        self.gmods.append( gmod.detach().to('cpu').squeeze().numpy() ) # output of convlstm
        self.grads.append( (grad / self.grad_mod._grad_norm).detach().to('cpu').squeeze().numpy() ) # input of convlstm (normalized)
                
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        
        ## store for visu
        self.varc.append( var_cost.detach().to('cpu').squeeze().numpy())
        self.pc.append( prior_term.detach().to('cpu').squeeze().numpy())
        self.obsc.append( obs_term.detach().squeeze().to('cpu').numpy() )
    
        return state - state_update

    def forward(self, batch):
        self.varc = []
        self.pc = [] 
        self.obsc = []
        self.grads = []
        self.gmods = []
        self.grad_loss = []
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            if not self.state_solver: 
                self.grad_mod.reset_state(batch.tgt)
            else: 
                self.grad_mod.reset_state(batch.tgt, batch.tgt)
            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)
            # if not self.training: # removed those two lines to do the same thing during training and testing time
            #    state = self.prior_cost.forward_ae(state)

        return state

    def zeros_init(self, x) : 
        state_init = torch.zeros_like(x)
        return state_init.to(device = next(self.parameters()).device) 
    
    def noise_init(self, x) : 
        rd_state = np.random.get_state()
        state_init = (1e-3*torch.randn(x.shape))
        np.random.set_state(rd_state)
        return state_init.to(device = next(self.parameters()).device) 
    
    def mnep_invop( self, mne_info, fwd, method = "MNE" ): 
        """  
        Compute the inverse operator for the minimum norm solution *method* (e.g MNE) based on the mne-python algorithms.
        intput : 
        - mne_info : mne-python *info* object associated with the eeg data
        - fwd : mne-python forward operator linked with the simulated data (head model)
        - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
        output : 
        - K : inverse operator (torch.tensor)
        """
        import mne
        from mne.minimum_norm.inverse import (_assemble_kernel,
                                              _check_or_prepare)
        ## compute a "fake" noise covariance
        random_state = np.random.get_state() # avoid changing all random number generation when using MNE init
        noise_eeg = mne.io.RawArray(
                np.random.randn(len(mne_info['chs']), 600), mne_info, verbose=False
            )
        noise_eeg, _ = mne.set_eeg_reference(noise_eeg, projection=True)
        np.random.set_state(random_state)
        noise_cov = mne.compute_raw_covariance(noise_eeg, verbose=False)
        ## compute the inverse operator (K)
        inv_op = mne.minimum_norm.make_inverse_operator(
            info=mne_info,
            forward=fwd,
            noise_cov=noise_cov,
            loose=0,
            depth=0,
            verbose=False
        )

        inv = _check_or_prepare(
            inv_op, 1, self.reg_mne, method ,None,False
        )
        
        K, _, _, _ = _assemble_kernel(
                inv, label=None, method=method, pick_ori=None, use_cps=True, verbose=False
            )
        return torch.from_numpy(K)
    
    def mnep_init(self, y, method="MNE"): 
        """  
        Inverse problem resolution : estimates x from y, using the inverse operator K (based on mne-python algorithms). 
        input : 
        - y : eeg data (batch, channel, time)
        - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
        """
        y = y.float()
        if self.inv_op is None: 
            self.inv_op = self.mnep_invop( self.mne_info, self.fwd.copy(), method ).float().to(device = next(self.parameters()).device)
        
        return torch.matmul(self.inv_op.to(device = next(self.parameters()).device), y)

## lightning module
class ESILitModule(pl.LightningModule):
    """  
    lightning module to put it all together and train the model
    """ 
    def __init__(self, solver, opt_fn, loss_fn, loss_ponde = {"gt":1.0, "prior":1/50}):
        ## grad_matrix = edge_grad, laplacian
        super().__init__()
        self.solver = solver
        self.opt_fn = opt_fn
        self.loss_fn = loss_fn    
        self.loss_ponde = loss_ponde


    def forward(self, batch):
        return self.solver(batch)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")[0]
    
    def step(self, batch, batch_idx, phase=""):
        # from contrib.eeg.utils_eeg import sigmoid_focal_loss
        out = self(batch=batch)
        ## loss with ponderation in parameters :
        loss_gt = self.loss_ponde["gt"] * self.loss_fn(batch.tgt, out) 
        loss_prior = self.loss_ponde["prior"] * self.loss_fn(out, self.solver.prior_cost.forward_ae(out))
        
        loss = loss_gt + loss_prior

        with torch.no_grad():
            self.log("model params mean", np.array([p.detach().cpu().mean() for p in self.solver.prior_cost.parameters()]).mean(), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_data_loss", self.loss_fn( batch.tgt, out ), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{phase}_ae_loss", self.loss_fn(out, self.solver.prior_cost.forward_ae(out)), prog_bar=False, on_step=False, on_epoch=True)

            # self.log(f"{phase}_reg_param", self.solver.reg_param.detach(), prog_bar=False, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

       
def optim_adam_gradphi( lit_mod, lr, lr_phi_frac=2 ): 
    """
    optimizer for both the grad model and the prior cost
    """
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr/lr_phi_frac}, # usually /2 - test 31 jan 2025
        ],
    )

def optim_adam_grad( lit_model, lr ):
    ## optimizer to use when the prior is pretrained and fixed during training
    return torch.optim.Adam(
        [
            {"params": lit_model.solver.grad_mod.parameters(), "lr": lr},
        ],
    )

## using learning rate scheduler
def cosanneal_lr_adam(lit_mod, lr, T_max=100, eta_min=1e-6) :#, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ]#, weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min),
    }
def triang_lr_adam(lit_mod, lr_min=1e-5, lr_max=1e-3, nsteps=100, lr_phi_frac=2):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / lr_phi_frac},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }

#################################
# give two different gradient parts # TG stands for "two gradients"
###################################""
class EsiGradSolverTG(EsiGradSolver) : 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def solver_step(self, state, batch, step):
        #with torch.backends.cudnn.flags(enabled=False):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        self.varc.append( var_cost.detach().to('cpu').squeeze().numpy())
        self.pc.append( self.prior_cost(state).detach().to('cpu').squeeze().numpy())
        self.obsc.append( self.obs_cost(state, batch).detach().squeeze().to('cpu').numpy() )
        prior_c = self.prior_cost(state)
        obs_c = self.obs_cost(state, batch)

        grad_obs = torch.autograd.grad(obs_c, state, create_graph=True)[0] # compute gradient of obs cost w.r.t state
        grad_prior = torch.autograd.grad(prior_c, state, create_graph=True)[0] # compute gradient of prior cost w.r.t state
        gmod = self.grad_mod( (grad_obs, grad_prior) ) 
        self.gmods.append( gmod.detach().to('cpu').squeeze().numpy() )
        self.grads.append( (grad_obs+grad_prior).detach().to('cpu').squeeze().numpy() )
                
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * (grad_prior+grad_obs)
        )
            
        return state - state_update

class ConvLstmGradModelTG(nn.Module): #TG stands for "two gradients"
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        # dim_in = 2*dim_in
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # self.conv_out = torch.nn.Conv2d(
        #     dim_hidden, dim_in//2, kernel_size=kernel_size, padding=kernel_size // 2
        # )
        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )
        ###
        # self.linear_comb = nn.Parameters(torch.tensor([0.5, 0.5])) #combine the two gradients
        ###
        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )


    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._grad_norm_obs = None
        self._grad_norm_prior = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        if self._grad_norm_obs is None:
            self._grad_norm_obs = (x[0]**2).mean().sqrt()
            self._grad_norm_prior = (x[1]**2).mean().sqrt()
        
        hidden, cell = self._state
        
#        x = torch.cat((x[0] / self._grad_norm_obs, x[1] / self._grad_norm_prior), dim=1)
        x = x[0] / self._grad_norm_obs + x[1] / self._grad_norm_prior
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out

### grad mod (convLSTM)
class RearrangedConvLstmGradModelTG(ConvLstmGradModelTG): #TG stands for "two gradients"
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        super().reset_state(inp)

    def forward(self, x):
        x = ( einops.rearrange(x[0], self.rearrange_bef), einops.rearrange(x[1], self.rearrange_bef) )
        x = super().forward(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x