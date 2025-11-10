"""
evaluate results on a different subject than the one used for training
"""

import csv
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from functools import partial
from pathlib import Path

import einops
import hydra
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from tqdm import tqdm
from omegaconf.omegaconf import open_dict

import contrib.eeg.utils_eeg as utl
from contrib.eeg import metrics as met
from contrib.eeg.data import EsiDatamodule
from contrib.eeg.model_short import (ConvAEPrior, CosineReshape,
                                     EsiBaseObsCost, EsiGradSolver,
                                     ESILitModule, RearrangedConvLstmGradModel,
                                     optim_adam_gradphi)
from contrib.eeg.models_directinv import HeckerLSTM, HeckerLSTMpl
from contrib.eeg.utils_eeg import (load_fwd, load_mne_info,
                                   load_model_from_conf, plot_source_estimate,
                                   plot_src_from_imgs, signal_to_windows,
                                   windows_to_signal)
from contrib.eeg import utils_eeg

home = os.path.expanduser('~')

parser = ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("-od", "--output_dir", type=str, help="name of the output directory", required=True)
parser.add_argument("-add_ovr", "--add_overrides", nargs="*", help="additional overrides")
parser.add_argument("-bsl", "--baselines", nargs="+", help="baselines neural network method to use", default=["1dcnn", "lstm"])
parser.add_argument("-bsl_conf", "--baseline_config", type=str, default="baselines_ses_retrained.yaml")

parser.add_argument("-test_ovr", "--test_overrides", nargs="*", help="test dataset overrides")
parser.add_argument("-i", "--eval_idx", type=int, help="index of data for visualisation", default=2)
parser.add_argument("-sv", "--surfer_view", type=str, default="lat", help="surfer view if different from the one in the default file")
parser.add_argument("-sh", "--show", action="store_true")
parser.add_argument("-tdc", "--test_data_config", type=str, help="test dataset config file", default='test_dataset.yaml')
parser.add_argument("-ott", "--on_train", action="store_true", help="on train dataset")
parser.add_argument("-nf", "--noise_fixed", action="store_true", help="use a dataset with fixed noise")
parser.add_argument("-m", "--method", type=str, help="method to evaluate", default="4dvar")
parser.add_argument("-subset", "--subset", type=str, help="which subset to use", default="left_back")
parser.add_argument("-sub", "--subset_name", type=str, default="left_back", help="name of the training subset to use")
parser.add_argument("-def_dataf", "--def_datafolder", action="store_true", help="Use default datafolder")
parser.add_argument("-sreord_bsl", "--source_reordering_bsl", action="store_true", help="Reorder the sources to match source indexing btwn source spaces")
parser.add_argument("-sreord_all", "--source_reordering_all", action="store_true", help="Reorder the sources to match source indexing btwn source spaces")
args = parser.parse_args()


pl.seed_everything(333)
device = torch.device("cpu")

output_dir = args.output_dir
config_path = Path( f"{args.output_dir}", ".hydra" )
default_datafolder = Path(home, "Documents/Data/simulation")


source_reordering_all = args.source_reordering_all
source_reordering_bsl = args.source_reordering_bsl
if source_reordering_bsl: 
    source_reordering_all = False 
if source_reordering_all: 
    source_reordering_bsl = True

# init hydra config
with initialize(config_path=str(config_path), version_base=None):
    cfg = compose(config_name="config.yaml")#, overrides=overr)

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

if 'datafolder' not in cfg or args.def_datafolder: 
    with open_dict(cfg) as d : 
        d.datafolder = default_datafolder

test_data_config = str(Path('config', 'dataset', args.test_data_config))
test_config = OmegaConf.load(test_data_config)
if args.test_overrides :
    for override in args.test_overrides : 
        arg_path, value = override.split("=") 
        arg_path_list = arg_path.split('.')
        
        current_config = test_config
        for arg in arg_path_list : 
            current_config = getattr(current_config, arg)
        value_type = type(current_config)
        OmegaConf.update( test_config, arg_path, value_type(value) )

datamodule_conf = {
    "dataset_kw": 
    {
        "datafolder":cfg.datafolder, 
        "simu_name": test_config.simu_name,
        "subject_name": test_config.subject_name,
        "source_sampling": test_config.source_sampling,
        "electrode_montage": test_config.electrode_montage,
        "orientation": test_config.orientation,
        "to_load": test_config.to_load,
        "snr_db": 5,
        "noise_type": {"white":1.},
        "scaler_type": 'linear_bis',
        "replace_root": True
    },
    "subset_name": args.subset_name,
    "per_valid": 1,
    "dl_kw":{
        "batch_size": 1, #16
    },
    # "time_window":args.time_window
}
cfg.dataset = test_config
cfg.datamodule = datamodule_conf

## also change the leadfield:
cfg.fwd.head_model_dict = {
    "subject_name": test_config.subject_name,
    "orientation": test_config.orientation,
    "electrode_montage": test_config.electrode_montage,
    "source_sampling": test_config.source_sampling
}
cfg.fwd.fwd_name = f'fwd_{test_config.source_sampling}-fwd.fif'  

cfg.mne_info.electrode_montage = test_config.electrode_montage

if args.on_train : 
    dm = hydra.utils.call(cfg.datamodule)
else : 
    dm = EsiDatamodule(**datamodule_conf)
     
dm.setup("test")
test_dl = dm.test_dataloader()

fwd = hydra.utils.call(cfg.fwd)
n_sources = fwd['sol']['data'].shape[1]
if source_reordering_bsl: 
    fs_dict = {
        "subject_name": "fsaverage",
        "orientation": "constrained",
        "electrode_montage": "standard_1020",
        "source_sampling": "ico3-spl-morph"
    }
    fwd_fs = utils_eeg.load_fwd( cfg.datafolder, fs_dict, "fwd_ico3-spl-morph-fwd.fif")
    print("##### REORDERING SOURCES #######")
    # @TODO : sauvegarder ces infos plutot que de les recalculer à chaque fois
    from scipy.spatial.distance import cdist 
    from scipy.optimize import linear_sum_assignment 
    import mne 
    ## load forward fsav: 
    spos_fsav = utils_eeg.get_src_position( fwd_fs['src'] )
    spos_sample = utils_eeg.get_src_position( fwd['src'] )
    distance_matrix = cdist(spos_fsav- spos_fsav.mean(0), spos_sample-spos_sample.mean(0))
    _, src_mapping = linear_sum_assignment(distance_matrix)
    inverse_mapping = np.argsort( src_mapping )
    ## change leadfield
    # fwd['sol']['data'] = fwd['sol']['data'][:,src_mapping]
    # mne.write_forward_solution( 
    #     Path(cfg.datafolder, 
    #         test_config.subject_name, test_config.orientation, test_config.electrode_montage, test_config.source_sampling, 
    #         "model", f'fwd_{test_config.source_sampling}-shuffled-fwd.fif'),
    #         fwd, overwrite=True
    # )
    if source_reordering_all:
        cfg.fwd.fwd_name = f'fwd_{test_config.source_sampling}-shuffled-fwd.fif'  
    # fwd = hydra.utils.call(cfg.fwd)
else: 
    src_mapping = np.linspace( 0, n_sources-1, n_sources ).astype(int)
    inverse_mapping = np.linspace( 0, n_sources-1, n_sources ).astype(int)

# mne_info = hydra.utils.call(cfg.mne_info)
leadfield = torch.from_numpy(fwd['sol']['data']).float()

### --- LOAD MODEL --- ###
import sys

model_path = Path( output_dir, "lightning_logs", "checkpoints", "best_ckpt.ckpt" )
if os.path.isfile(model_path): 
    print("Model exists \U0001F44D")
else: 
    print("try other model path")
    print(model_path)
    sys.exit()

litmodel = hydra.utils.call(cfg.litmodel)

litmodel.solver.prior_cost.to(device=device)
if hasattr( litmodel.solver.prior_cost, "edge_index" ): 
    litmodel.solver.prior_cost.edge_index = litmodel.solver.prior_cost.edge_index.to(device)

loaded_mod = torch.load(model_path, map_location=torch.device('cpu'))
litmodel.load_state_dict( loaded_mod['state_dict'] )
litmodel.eval()

### --- LOAD BASELINES --- ###
baselines = args.baselines
if baselines[0] == "none": 
    baselines = []
    baseline_nets = []
else : 
    baseline_conf_path = args.baseline_config
    baseline_nets = dict(zip(baselines, []*len(baselines)))
    baseline_config = OmegaConf.load(str(Path("baselines", baseline_conf_path)))
    # baseline_config =  OmegaConf.load(args.baseline_config)
    for bsl in baselines: 
        baseline_nets[bsl] = load_model_from_conf(bsl, baseline_config)

if args.on_train :
    figs_path =  Path(output_dir, "metrics", "train")
else : 
    figs_path =  Path(output_dir, "metrics", "test")

os.makedirs( figs_path , exist_ok=True)

mne_info = load_mne_info( electrode_montage = test_config.electrode_montage, sampling_freq=512 )
fs = np.floor(mne_info['sfreq'])
n_times = test_config.n_times
t_vec = np.arange(0, n_times / fs, 1 / fs)


spos = torch.from_numpy(fwd['source_rr'])


neighbors = utl.get_neighbors(
    [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
    [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
)
linear_methods = ["MNE", "sLORETA"]
# nn_methods = ["4dvar"] + baselines
# methods = ["gt"] + nn_methods + linear_methods 
nn_methods = baselines + ["4dvar"]
methods = ["gt"] + linear_methods + nn_methods
# methods = ["gt"] + [args.method]
########################### REORDER SOURCES ##########################################

from scipy.spatial.distance import cdist 
from scipy.optimize import linear_sum_assignment 
import mne 
from contrib.eeg import utils_eeg
## load forward fsav: 
## load fsaverage fwd object
fs_dict = {
    "subject_name": "fsaverage",
    "orientation": "constrained",
    "electrode_montage": "standard_1020",
    "source_sampling": "ico3-spl-morph"
}
fwd_fs = utils_eeg.load_fwd( cfg.datafolder, fs_dict, "fwd_ico3-spl-morph-fwd.fif")

spos_fsav = utils_eeg.get_src_position( fwd_fs['src'] )
spos_sample = utils_eeg.get_src_position( fwd['src'] )
distance_matrix = cdist(spos_fsav- spos_fsav.mean(0), spos_sample-spos_sample.mean(0))
_, src_mapping = linear_sum_assignment(distance_matrix)
inverse_mapping = np.argsort( src_mapping )

## change leadfield ? ## 
# add a condition
######################################################################################
##### eval #####
n_val_samples = len(dm.test_ds)

nmse_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
loc_error_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
psnr_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
time_error_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
auc_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
cosine_sim_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
cosine_ae_sim_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
loc_error_n_dict = {method: np.empty((n_val_samples, 1)) for method in methods}

overlapping_regions = 0
coss = CosineReshape()
for k in tqdm(range(n_val_samples)):
    eeg_gt, src_gt = dm.test_ds[k]
    eeg_gt, src_gt = eeg_gt.float().clone(), src_gt.float().clone()

    eeg_gt_unscaled = eeg_gt.clone() * dm.test_ds.max_eeg[k]
    src_gt_unscaled = src_gt.clone() * dm.test_ds.max_src[k]
    #####  
    # data covariance:
    # activity_thresh = 0.1
    # noise_cov, data_cov, nap = inv.mne_compute_covs(
    #    (M_unscaled).numpy(), mne_info, activity_thresh
    # )
    ### TEST BETTER NOISE COV
    raw_noise = mne.io.RawArray(
        data=np.random.randn(cfg.dataset.n_electrodes, 600),
        info=mne_info,verbose=False
    )
    noise_cov = mne.compute_raw_covariance(raw_noise, verbose=False)   

    raw_eeg = mne.io.RawArray(
        data=eeg_gt_unscaled, info=mne_info, first_samp=0.0, verbose=False
    )
    rraw_eeg = mne.set_eeg_reference(raw_eeg, "average", projection=True, verbose=False)[0]

    stc_hat = dict(zip(methods, [None] * len(methods)))
    stc_hat["gt"] = src_gt_unscaled.numpy()
    lambda2 = 1.0 / (cfg.datamodule.dataset_kw.snr_db**2)
    ###
    for m in methods:
        
        if m in linear_methods:
            inv_op = mne.minimum_norm.make_inverse_operator(
                info=raw_eeg.info,
                forward=fwd,
                noise_cov=noise_cov,
                loose=0,
                depth=0,
                verbose=False
            )
            stc_hat[m] = mne.minimum_norm.apply_inverse_raw(
                raw=raw_eeg, inverse_operator=inv_op, lambda2=lambda2, method=m, verbose=False
            )
            stc_hat[m] = stc_hat[m].data
        
        elif m in baselines : 
            with torch.no_grad():
                batch = TrainingItem(
                    input= eeg_gt.float().unsqueeze(0), 
                    tgt= src_gt.float().unsqueeze(0)) 
                stc_hat[m] = baseline_nets[m](batch.input).detach().squeeze()  

            stc_hat[m] = stc_hat[m].detach().numpy() * dm.test_ds.max_src[k].numpy()
            if source_reordering_bsl: 
                stc_hat[m] = stc_hat[m][inverse_mapping,:]

        elif m == "4dvar":
            batch = TrainingItem(
                input= eeg_gt.float().unsqueeze(0).clone(), 
                tgt= src_gt.float().unsqueeze(0).clone()) 

            with torch.no_grad():
                output = litmodel(batch).detach()
                output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(batch) )
                if source_reordering_all:
                    output = output[:,inverse_mapping,:]
                    output_ae = output_ae[:,inverse_mapping,:]
            stc_hat[m] = output.squeeze()
            stc_hat[m] = stc_hat[m].detach().numpy() * dm.test_ds.max_src[k].numpy()
            

#------------------------------------------------------------------------------------------------------------------------#
        src_hat = torch.from_numpy(stc_hat[m])
        le = 0
        le_n = 0
        te = 0
        nmse = 0
        auc_val = 0
        seeds_hat = []
        ## check for overlap ------ @TODO : fix ok for 2 sources, not for more
        seeds = dm.test_ds.md[k]["seeds"]
        if type(seeds) is int:
            seeds = [seeds]

        patches = [ [] for _ in range(len(seeds)) ]
        for kk in range(len(seeds)) : 
            patches[kk] = dm.test_ds.md[k]['act_src'][f'patch_{kk+1}'] 
        if len(seeds) > 1:
            inter = list( 
                set(patches[0]).intersection(patches[1])
            )
            if len(inter)>0 : # for overlapping regions : only keep seed with max activity
                overlapping_regions += 1
                to_keep = torch.argmax( torch.Tensor([src_gt[seeds[0], :].abs().max(), src_gt[seeds[1], :].abs().max() ]) )
                seeds = [ seeds[to_keep] ]
        ## ----------------------
        act_src = [ s for l in patches for s in l ]
            # compute metrics -----------------------------------------------------------------
        for kk in range(len(seeds)) :
            s = seeds[kk]
            other_sources = np.setdiff1d(
                act_src, patches[kk]
            ) # source from other patches
            t_eval_gt = torch.argmax(src_gt[s, :].abs())
                # find estimated seed, in a neighboring area
            eval_zone = utl.get_patch(order=7, idx=s, neighbors=neighbors)
            ## remove sources from other patches of the eval zone (case of close sources regions) ##
            eval_zone = np.setdiff1d(eval_zone, other_sources)

            # find estimated seed, in a neighboring area
            # eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)
            s_hat = eval_zone[torch.argmax(src_hat[eval_zone, t_eval_gt].abs())]

            t_eval_pred = torch.argmax(src_hat[s_hat, :].abs())

            le += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())

            le_n+= utl.le_neighb(fwd, neighbors, s, src_gt, src_hat) 

            te += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred])
            auc_val += met.auc_t(
                src_gt_unscaled, src_hat, t_eval_gt, thresh=True, act_thresh=0.0
            )  # probablement peut mieux faire

            #nmse += met.nmse_t_fn(j_unscaled, j_hat, t_eval_gt)
            nmse_tmp = ( (
                src_gt_unscaled[:,t_eval_gt] / src_gt_unscaled[:,t_eval_gt].abs().max() - src_hat[:,t_eval_gt] / src_hat[:,t_eval_gt].abs().max() 
                )**2 ).mean()
            nmse += nmse_tmp
                
            seeds_hat.append(s_hat)

        le = le / len(seeds)
        le_n = le_n /len(seeds)
        te = te / len(seeds)
        nmse = nmse / len(seeds)
        auc_val = auc_val / len(seeds)
        tmaxs_pred = torch.argmax(src_hat[seeds_hat, :].abs(), dim=1)
        # time error (error on the instant of the max. activity):
        time_error_dict[m][k] = te
        # print(f"time error: {time_error*1e3} [ms]")

        # localisation error
        loc_error_dict[m][k] = le*1e3
        loc_error_n_dict[m][k] = int(le_n)
        # print(f"localisation error: {loc_error*1e3} [mm]")

        # instant nMSE:
        nmse_dict[m][k] = nmse
        # print(f"nmse at instant of max activity: {nmse_t:.4f}")

        # PSNR
        psnr_dict[m][k] = psnr(
            (src_gt_unscaled / src_gt_unscaled.abs().max()).numpy(),
            (src_hat / src_hat.abs().max()).numpy(),
            data_range=(
                (src_gt_unscaled / src_gt_unscaled.abs().max()).min()
                - (src_hat / src_hat.abs().max()).max()
            ),
        )
        # print(f"psnr for total source distrib: {psnr_val:.4f} [dB]")

        # AUC
        # act_src = esi_datamodule.val_ds.act_src[k]
        auc_dict[m][k] = auc_val
        # print(f"auc: {auc_val:.4f}")

        # cosine similarity
        cosine_sim_dict[m][k] = coss( src_gt_unscaled.unsqueeze(0), src_hat.unsqueeze(0))
        if m == "4dvar":
            cosine_ae_sim_dict[m][k] =  coss( src_gt_unscaled.unsqueeze(0), output_ae)
        # change plots to visu. multiple sources
        idx_max_gt = seeds[0]
        idx_max_pred = seeds_hat[0]
import os

#####################################################################
#############################################################################
import pandas as pd

os.makedirs(Path(output_dir, "evals", "test", args.test_data_config), exist_ok=True)
## first : save every value, for each metric and each method : 
metrics = {
    "LE":loc_error_dict, 
    "nMSE":nmse_dict, 
    "AUC":auc_dict, 
    "PSNR":psnr_dict, 
    "TE":time_error_dict
    }
for me in metrics :
    list_of_arrays = [metrics[me][m].squeeze() for m in metrics[me].keys()]
    df = pd.DataFrame(data = list_of_arrays).T
    df.columns = list(metrics[me].keys())

    df.to_csv(Path(output_dir, "evals", "test", args.test_data_config,  f"{me.upper()}.csv"), index=False)

 
# save mean and std value
data = {
    'metric': list(metrics.keys())
    }
for method in methods: 
    data.update( 
        {method: [metrics[metric][method].mean() for metric in metrics.keys()]}
)
df = pd.DataFrame(data=data)
df = df.set_index('metric')
df.to_csv(Path(output_dir, "evals", "test",args.test_data_config, "MEANS.csv"), float_format='%.4f')
## std
data = {
    'metric': list(metrics.keys())
    }
for method in methods: 
    data.update( 
        {method: [metrics[metric][method].std() for metric in metrics.keys()]}
)
df = pd.DataFrame(data=data)
df = df.set_index('metric')
df.to_csv(Path(output_dir, "evals", "test", args.test_data_config, "STDS.csv"), float_format='%.4f')

for method in methods:
    print(f" >>>>>>>>>>>>>>> Results method {method} <<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"mean localisation error: {loc_error_dict[method].mean():.2f} [mm]")
    print(f"mean localisation neighbors error: {loc_error_n_dict[method].mean():.2f} [mm]")
    print(f"mean AUC: {auc_dict[method].mean()*100:.2f}")
    print(f"mean nMSE at instant of max activity: {nmse_dict[method].mean():.4f}")
    print(f"mean PSNR for total source distrib: {psnr_dict[method].mean():.2f} [dB]")
    print(f"mean time error: {time_error_dict[method].mean()*1e3:.2f} [ms]")
    print("-------------------------")
    print(f"mean AE cosine sim: {cosine_ae_sim_dict[method].mean():.3f} [ms]")
    
