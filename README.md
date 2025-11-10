update: 2025/10/15
Code for model-based EEG source imaging following the work of Fablet et al. with codes in the following github: https://github.com/CIA-Oceanix/4dvarnet-starter

# Structure
- `src`: folder with original source code for 4DVarNet approach
- `contrib/eeg`: code specific to the application of the method to EEG source imaging
- `config`: hydra configuration files
- `config_baselines`: configuration files for the direct inversion baseline models
- `pretrained_priors`: pretrained prior models
- `trained_baselines`: baseline direct inversion models

RK: With hydra by default the results of a training will be saved in the folder `outputs`, in a subfolder build with the date and time of training. /!\ When used on Jean Zay -> sometimes multiple training are launched at the same time so add an argument to the training command line to add a suffix to the folder name in order to create different folder for the same date/time. 

If multirun is used the results are saved in `multirun` in a subfolder build with the date and time of training, with different number of subfolder for each run. 


# Data
For SEREEGA data: different way to create the dataset
- EsiDataset -> base dataset using TrainingItem as output for getitem
- EsiDatasetNoiseFixed -> Load EEG data simulated with a fixed noise (instead of adding noise when loading data)
- EsiDatasetNoiseSrc -> Add noise to source data (test with cosine similarity)
- EsiDatasetTw -> Load only a small time window of the data
- EsiDatasetAE -> Dataset to train autoencoder (output source+noise and source)
- EsiDatasetID -> Dataset to train direct inversion methods (do not use TrainingItem)

-> build datamodule given an option of dataset to use.

*TODO*: add neural mass models datasets utilisation.

# Model
- Obs_cost: observation cost: $f_o(y, Hx)$ -> given the forward object and a loss function computes the observation cost
- Prior_cost: autoencoder model, two "forward" functions -> forward which computes $f_p(x, \phi(x))$, given a loss function $f_p$, 
    and "forward_ae" which computes $\phi(x)$ the output of the autoencoder.
    Default = 1D-conv autoencoder
- Grad_mod: convLSTM by default. The model which learns the optimization (takes gradient as input, output state update)
-> Obs_cost, prior_cost and grad_mod used to build a forward pass in solver:
- Solver: 
    - initialization of the state (mne, zeros, noise)
    - for n_step steps: compute the variational cost, compute the gradient w.r.t the state (using autograd), update the state with $\psi(grad)$. 
    - returns the state at the last iteration

=> training logic, loss computation... in the lightning module

# Training
Training the base model with a convolution autoencoder and convLSTM solver: 
```
python train_from_config.py suffix=demo
```
Should create a folder `outputs/<date>/<time>_demo`, in which :
- `.hydra` contains the configuration file, and the overrides. 
- `lightning_logs/checkpoints` contains the checkpoints saved during training
- `metrics.csv`contains the information logged during training (loss function, ...)

Examples of overrides to change study initialization: 
```
python train_from_config.py suffix=inoise litmodel.solver.init_type=noise

python train_from_config.py suffix=izeros litmodel.solver.init_type=zeros

```


Examples of overrides to study loss function: 
```

python train_from_config.py suffix=mse cost_functions=mse_costs 

# with MSE it is a good idea to increase the regularisation parameter in the variational cost, as well as the loss ponderations: 
python train_from_config.py suffix=mse_ponde cost_functions=mse_costs litmodel.solver.reg_param.prior=1e3 litmodel.loss_ponde.gt=1e3 litmodel.loss_ponde.prior=20 

```

Example: training on the whole dataset VS sub-dataset

```
python train_from_config.py suffix=demo datamodule.subset_name="left_back"
# or for the whole dataset: 
python train_from_config.py suffix=demo datamodule.subset_name=none

```

## convAE + convLSTM
```
python train_from_config.py suffix=conv-conv
```


# Graphs: 
Codes related to the use of graph neural networks in the frameworks are in `contrib/eeg/graphs.py`. The codes are based on the use of pytorch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/

**Remark**: training with graph neural networks takes quite some time, among other things because at each forward pass the batch data is converted to a graph batch data using a for loop -> *todo* = optimize this process. Also the graph unet requires a lot of memory and no work was done in order to improve the models. 

Configuration to use graphs: `config_gae_var.yaml`
## GAE + convLSTM
```
python train_from_config.py --config-name=config_gae_var suffix=gae_clstm
``` 

## GAE + GAE
```
python train_from_config.py --config-name=config_gae_var suffix=gae_gae grad_mod=gae_gradmod
``` 

## GAE + gLSTM
```
python train_from_config.py --config-name=config_gae_var suffix=gae_glstm grad_mod=glstm_gradmod
``` 

# Model evaluation: 

For a model saved in : `outputs/2025-10-15/15-35-05_default`, on the base test dataset:
```
python eval_results.py -od=outputs/2025-10-15/15-35-05_default -bsl_conf=baselines_ses_125ms_cosine.yaml -tdc=base_test_dataset.yaml
```
Computes the localisation error, the neighbor localisation error, the nMSE, PSNR, AUC, time error and some additional metrics based on cosine similarity. Save the information in the output folder of the trained model in `evals`

# Result visualization
```
python -i visu_results.py -od=outputs/2025-10-15/15-35-05_default -bsl_conf=baselines_ses_125ms_cosine.yaml -tdc=base_test_dataset.yaml -i=0
```

# Inter-subject experiment: 
-> change the configuration file for the test dataset in eval_results, as well as baselines if not available

-> specific script for evaluation -> `eval_results_inter_subject_reorder.py` -> allow to evaluate results for inter subject with and without source reordering.
-> specific script for visualization -> `visu_results_inter_subject_reorder.py`  
**a word on source reordering**: 
The two source spaces used do not match in terms of indexing, which decreases the performances of the models based on convolutions/linear indexing of the sources. 
For a fairer evaluation, source reordering can be applied. Source reordering is based on linear assignement in order to find a bijection between the two sources spaces ordering, which matches closest sources in terms of euclidean distances.

-> for visualization: results are morphed back to fsaverage to have coherence between visualizations. 

# Extract a specific sub-dataset:
Some experiments in this work where performed only on 1/4 of the brain (test some models while decreasing the training time before training on the whole dataset, see generalization to the whole brain from only a sub-part of the cortex...)
-> For SEREEGA data: the `sub_dataset.ipynb` notebook allows to create a file giving the list of data corresponding to a specific condition on the position of the maximum source ("left-back", "right-back", "left-front" or "right-front"). This text file is then saved in the corresponding simulation folder to be used when loading the dataset.


# References
## Data simulation
- SEREEGA: 
- Extension simulation:
## Direction inversion: 
- LSTM : 
- 1D-CNN: 
## 4DVarNet: 


# For the article
Results are in the folder: `outputs_chpt4`. The names of the subfolder are built with `subset_init_phi_psi`. 
For example 
- training on the whole brain, init MNE and phi = convAE, psi=convAE -> `whole_mne_conv_conv`
- training on left_back, init MNE, phi=GAE, psi=convLSTM -> `quart_mne_gae_conv`
