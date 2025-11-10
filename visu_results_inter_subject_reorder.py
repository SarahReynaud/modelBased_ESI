"""
Same as the notebook "visu_results.ipynb" but in a python script.
-> visualize results given an output folder
"""

import csv
import os
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
from scipy.io import loadmat
from torch import nn
from omegaconf.omegaconf import open_dict

from contrib.eeg.data import EsiDatamodule, EsiDataset
from contrib.eeg.model_short import (ConvAEPrior, EsiBaseObsCost,
                                     EsiGradSolver, ESILitModule,
                                     RearrangedConvLstmGradModel,
                                     optim_adam_gradphi)
from contrib.eeg.models_directinv import HeckerLSTM, HeckerLSTMpl
from contrib.eeg.utils_eeg import (load_fwd, load_mne_info,
                                   load_model_from_conf, plot_source_estimate,
                                   plot_src_from_imgs, signal_to_windows,
                                   windows_to_signal, windows_to_signal_center)
from contrib.eeg import utils_eeg

home = os.path.expanduser('~')
anatomy_folder = Path(f"{home}/Documents/deepsif/DeepSIF-main/anatomy")


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
parser.add_argument("-tw", "--time_window", action="store_true", help="time window")
parser.add_argument("-ovlp", "--overlap", type=int, default=7, help="overlap for the time window")
parser.add_argument("-wl", "--window_length", type=int, default=7, help="length of time window")
parser.add_argument("-ott", "--on_train", action="store_true", help="on train dataset")
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

overlap=args.overlap
window_length = args.window_length
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
# first load orginal forward
fs_dict = {
    "subject_name": "fsaverage",
    "orientation": "constrained",
    "electrode_montage": "standard_1020",
    "source_sampling": "ico3-spl-morph"
}
fwd_fs = utils_eeg.load_fwd( cfg.datafolder, fs_dict, cfg.fwd.fwd_name)
# change config
cfg.fwd.head_model_dict = {
    "subject_name": test_config.subject_name,
    "orientation": test_config.orientation,
    "electrode_montage": test_config.electrode_montage,
    "source_sampling": test_config.source_sampling
}
cfg.fwd.fwd_name = f'fwd_{test_config.source_sampling}-fwd.fif'  
# load sample fwd

fwd = hydra.utils.call(cfg.fwd)
if "mne_info" in cfg: 
    cfg.mne_info.electrode_montage = test_config.electrode_montage

if args.on_train : 
    dm = hydra.utils.call(cfg.datamodule)
else : 
    dm = EsiDatamodule(**datamodule_conf)
     
dm.setup("test")
test_dl = dm.test_dataloader()

### compute index permutation ###
n_sources = fwd['sol']['data'].shape[1]
if source_reordering_bsl: 
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

################################################
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

figs_path = Path(output_dir, "figs", "intersubj")
os.makedirs( figs_path , exist_ok=True)

idx = args.eval_idx
idx_v = idx
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt']) # batch format for 4DVARNET
eeg, src = dm.test_ds[idx_v]
batch = TrainingItem(input=eeg.unsqueeze(0), tgt=src.unsqueeze(0))
idx_v = 0

## Visu the GT data
plt.figure()
plt.subplot(121)
plt.imshow(batch.input[idx_v,:,:].squeeze().numpy())
plt.colorbar()
plt.axis('off')
plt.title('Input:EEG data')
plt.subplot(122)
plt.imshow(batch.tgt[idx_v,:,:].squeeze().numpy())
plt.axis('off')
plt.colorbar()
plt.title('Output: Source activity')
plt.savefig(Path(figs_path, f"gt_data_{idx}.png"))
if args.show :
    plt.show(block=False)    
else : 
    plt.close()

plt.figure(figsize=(10,5))
plt.subplot(121)
for e in range(test_config.n_electrodes):
    plt.plot(batch.input[idx_v,e,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('EEG data')
plt.subplot(122)
for s in range(test_config.n_sources): 
    plt.plot(batch.tgt[idx_v,s,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('Source activity')
plt.tight_layout()
plt.savefig(Path(figs_path, f"gt_data_waveform_{idx}.png"))
if args.show :
    plt.show(block=False)    
else : 
    plt.close()

overlap=args.overlap
window_length = args.window_length
to_keep = 4
# print(overlap)
with torch.no_grad():
    if args.time_window: 
        windows_input = signal_to_windows(batch.input, window_length=window_length, overlap=overlap, pad=True) 
        windows_tgt = signal_to_windows(batch.tgt, window_length=window_length, overlap=overlap, pad=True) 
        windows = TrainingItem(input=windows_input.squeeze(), tgt=windows_tgt.squeeze())
        output = litmodel(windows)
        # output = windows.tgt
        output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(windows) ) # check the output of the prior model
        output = torch.from_numpy( windows_to_signal(output.unsqueeze(1), overlap=overlap, n_times=batch.input.shape[-1]) )
        output_ae = torch.from_numpy( windows_to_signal(output_ae.unsqueeze(1), overlap=overlap, n_times=batch.input.shape[-1]) )
    else :    
        output = litmodel(batch).detach()
        output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(batch) ) # check the output of the prior model
        output_ae_gt = litmodel.solver.prior_cost.forward_ae( batch.tgt ).detach()
    if source_reordering_all:
        output = output[:,inverse_mapping,:]
        output_ae = output_ae[:,inverse_mapping,:]
        output_ae_gt = output_ae_gt[:,inverse_mapping,:]

if baselines: 
    with torch.no_grad(): 
        lstm_output = baseline_nets["lstm"](batch.input)
        cnn_output = baseline_nets["1dcnn"](batch.input)
    if source_reordering_bsl: 
        lstm_output = lstm_output[:,inverse_mapping,:]
        cnn_output = cnn_output[:,inverse_mapping,:]

# from scipy.ndimage import gaussian_filter1d
batch = TrainingItem( input=eeg.unsqueeze(0), tgt=src.unsqueeze(0))
plt.figure(figsize=(10,5))
plt.subplot(121)
for s in range(test_config.n_sources):
    plt.plot(batch.tgt[idx_v,s,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('source GT data')
plt.subplot(122)
for s in range(test_config.n_sources): 
    # plt.plot(gaussian_filter1d( output[idx_v,s,:].squeeze().numpy(), sigma=0.8))
    plt.plot(output[idx_v,s,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('Source estimated')
plt.tight_layout()
plt.savefig(Path(figs_path, f"gt_and_hat_source_data_waveform_{idx}.png"))
if args.show :
    plt.show(block=False)    
else : 
    plt.close()
# sys.exit()



## MNE and sLORETA, using mne-python implementation
mne_info = load_mne_info( electrode_montage = test_config.electrode_montage, sampling_freq=512 )
raw_eeg = mne.io.RawArray(batch.input[idx_v,:,:].squeeze().numpy(), mne_info)
raw_eeg.set_eeg_reference(projection=True, verbose=False)
noise_eeg = mne.io.RawArray(
        np.random.randn(batch.input[idx_v,:,:].squeeze().numpy().shape[0], 500), mne_info, verbose=False
    )
noise_cov = mne.compute_raw_covariance(noise_eeg, verbose=False)
lambda2 = 1.0 / (test_config.snr_db**2) ## !! this could be tuned to improve the results

inv_op = mne.minimum_norm.make_inverse_operator(
    info=raw_eeg.info,
    forward=fwd,
    noise_cov=noise_cov,
    loose=0,
    depth=0,
    )

m = "MNE"
stc_mne = mne.minimum_norm.apply_inverse_raw(
    raw=raw_eeg, inverse_operator=inv_op, lambda2=lambda2, method=m
)

m = "sLORETA"
stc_slo = mne.minimum_norm.apply_inverse_raw(
    raw=raw_eeg, inverse_operator=inv_op, lambda2=lambda2, method=m
)

mne_output = stc_mne.data
slo_output = stc_slo.data

## eval time / visu time : max activity
t_max = np.argmax( batch.tgt[idx_v,:,:].squeeze().abs().sum(0).numpy() )

######
# 24 oct
normalised_output = output.squeeze()[:,t_max] / output.abs().squeeze()[:,t_max].max()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(fwd['source_rr'][:,0], fwd['source_rr'][:,1], fwd['source_rr'][:,2], c=normalised_output, marker='o')
plt.show(block=False)

#####
# compute metrics on the given data
from contrib.eeg import metrics as met
from contrib.eeg import utils_eeg as utl

seeds = dm.test_ds.md[idx]["seeds"]
if type(seeds) is int:
    seeds = [seeds]
s = seeds[0]

if test_config.source_sampling == "fsav_994": 
    neighbors = np.squeeze( loadmat(f"{anatomy_folder}/fs_cortex_20k_region_mapping.mat")['nbs'] )
    # reshape neighbors
    l_max           = np.max( np.array([len(l[0]) for l in neighbors]) )
    neighb_array    = np.zeros( (len(neighbors), l_max) )
    for i in range(len(neighbors) ) : 
        l = neighbors[i][0]
        neighb_array[i,:len(l)] = l -1
        if len(l)<l_max: 
            neighb_array[i,len(l):] = None 
    neighb_array = neighb_array.astype(np.int64)

    neighbors = neighb_array.copy() 
    del neighb_array
else : 
    neighbors = utl.get_neighbors(
        [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
        [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
)


# def le_and_auc(fwd, neighbors, s, src, src_hat): 
#     src_hat = torch.from_numpy(src_hat)
#     spos = torch.from_numpy(fwd['source_rr'])
#     t_eval_gt = torch.argmax(src[s, :].abs())
#     # find estimated seed, in a neighboring area
    
#     eval_zone = utl.get_patch(order=20, idx=s, neighbors=neighbors)
#     # find estimated seed, in a neighboring area
#     s_hat = eval_zone[torch.argmax(src_hat[eval_zone, t_eval_gt].abs())]
#     le = torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
#     auc_val = met.auc_t(
#         src, src_hat, t_eval_gt, thresh=True, act_thresh=0.0
#         )  # probablement peut mieux faire
#     return le*1e3, auc_val #le in mm
def le_and_auc(fwd, neighbors, s, src, src_hat): 
    src_hat = torch.from_numpy(src_hat)
    spos = torch.from_numpy(fwd['source_rr'])
    t_eval_gt = torch.argmax(src[s, :].abs())
    # estimated source = source with max energy: 
    s_hat = torch.argmax(src_hat.sum(1).abs())
    ## is the GT source in the first order neighborhood of the s_hat source?
    le = -1
    o = 0
    while le < 0 and o < 10:
        neighb = utl.get_patch(order=o, idx=s_hat, neighbors=neighbors)
        if s in neighb: 
            le = o
        o += 1
    
    # find estimated seed, in a neighboring area
    # s_hat = eval_zone[torch.argmax(src_hat[eval_zone, :].sum(1).abs())]
    
    # eval_zone = np.delete(eval_zone, torch.argmax(src_hat[eval_zone, :].sum(1).abs()))
    # s_hat = eval_zone[torch.argmax(src_hat[eval_zone, :].sum(1).abs())]
    # le = torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
    auc_val = met.auc_t(
        src, src_hat, t_eval_gt, thresh=True, act_thresh=0.0
        )  # probablement peut mieux faire
    return le, auc_val #le in mm

# fwd['sol']['data'] = fwd['sol']['data'][:,inverse_mapping] ## reorder
le_mb, auc_mb = le_and_auc(fwd, neighbors, s, src, output[idx_v,:,:].detach().numpy() )
le_mne, auc_mne = le_and_auc(fwd, neighbors, s, src, mne_output )
le_slo, auc_slo = le_and_auc( fwd, neighbors, s, src, slo_output )
if baselines: 
    le_lstm, auc_lstm = le_and_auc(fwd, neighbors, s, src, lstm_output[idx_v,:,:].numpy() )
    le_cnn, auc_cnn = le_and_auc(fwd, neighbors, s, src, cnn_output[idx_v,:,:].numpy() )

###
from contrib.eeg.utils_eeg import plot_source_estimate_sub, plot_source_estimate_morph
# img_gt = plot_source_estimate_sub(
#     src=batch.tgt[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
# img_mb = plot_source_estimate_sub(
#     src=output[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
# img_mb_ae = plot_source_estimate_sub(
#     src=output_ae[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
# img_mb_ae_gt = plot_source_estimate_sub(
#     src=output_ae_gt[idx_v,:,:][inverse_mapping,:].numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)

# if baselines: 
#     img_lstm = plot_source_estimate_sub(
#         src=lstm_output[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
#     img_cnn = plot_source_estimate_sub(
#         src=cnn_output[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)

# img_mne = plot_source_estimate_sub(src=mne_output, t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
# img_slo = plot_source_estimate_sub(src=slo_output, t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)

# # visu GT, output from 4dvarnet and phi(output) : check if the prior autoencoder is working / learned
# if baselines: 
#     plot_src_from_imgs({"GT":img_gt, "MB": img_mb, "phi(MB)": img_mb_ae, "phi(GT)": img_mb_ae_gt}, ["GT","MB", "phi(MB)", "phi(GT)"])
#     plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
#     if args.show: 
#         plt.show(MBock=False)
#     else :
#         plt.close()

#     plot_src_from_imgs({"GT":img_gt, "MB": img_mb, "LSTM":img_lstm, "1DCNN":img_cnn}, ["GT","MB", "LSTM", "1DCNN"])
#     plt.savefig(Path(figs_path, f"cortex_learning_based_idx_{idx}.png"))
#     if args.show: 
#         plt.show(MBock=False)
#     else :
#         plt.close()

#     plot_src_from_imgs({"GT":img_gt, "MB": img_mb, "LSTM":img_lstm, "1DCNN":img_cnn, "MNE": img_mne, "sLORETA":img_slo}, 
#                        ["GT","MB", "LSTM", "1DCNN", "MNE", "sLORETA"], 
#                        subtitles=["\n0 -1"] + [f"\n{le:.3f} - {auc*100:.2f}" for le,auc in [(le_mb, auc_mb), (le_lstm, auc_lstm), (le_cnn, auc_cnn), (le_mne, auc_mne), (le_slo, auc_slo)]])
#     plt.savefig(Path(figs_path, f"cortex_all_methods_{idx}.png"))
#     if args.show: 
#         plt.show(MBock=False)
#     else :
#         plt.close()

#     plot_src_from_imgs({"MNE": img_mne, "sLORETA":img_slo}, ["MNE", "sLORETA"])
#     plt.savefig(Path(figs_path, f"cortex_nl_based_idx_{idx}.png"))
#     if args.show: 
#         plt.show(MBock=False)
#     else :
#         plt.close()

# else : 
#     plot_src_from_imgs({"GT":img_gt, "MB": img_mb, "phi(MB)": img_mb_ae, "phi(GT)": img_mb_ae_gt}, ["GT","MB", "phi(MB)", "phi(GT)"])
#     plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
#     if args.show: 
#         plt.show(MBock=False)
#     else :
#         plt.close()
    
#     plot_src_from_imgs({"GT":img_gt, "MB": img_mb, "MNE": img_mne, "sLORETA":img_slo}, 
#                        ["GT","MB", "MNE", "sLORETA"], 
#                        subtitles=["\n0 -1"] + [f"\n{le:.3f} - {auc*100:.2f}" for le,auc in [(le_mb, auc_mb), (le_mne, auc_mne), (le_slo, auc_slo)]])
#     plt.savefig(Path(figs_path, f"cortex_all_methods_{idx}.png"))
#     if args.show: 
#         plt.show(MBock=False)
#     else :
#         plt.close()

### WITH MORPHING BACK TO FSAVERAGE
img_gt_morph = plot_source_estimate_morph(
    src=batch.tgt[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd, fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
img_mb_morph = plot_source_estimate_morph(
    src=output[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd, fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
img_mb_ae_morph = plot_source_estimate_morph(
    src=output_ae[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd, fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
img_mb_ae_gt_morph = plot_source_estimate_morph(
    src=output_ae_gt[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd, fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)

if baselines: 
    img_lstm_morph = plot_source_estimate_morph(
        src=lstm_output[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd,fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
    img_cnn_morph = plot_source_estimate_morph(
        src=cnn_output[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd, fwd_fs=fwd_fs,fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)

img_mne_morph = plot_source_estimate_morph(
    src=mne_output, t_max=t_max, fwd=fwd,fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)
img_slo_morph = plot_source_estimate_morph(
    src=slo_output, t_max=t_max, fwd=fwd, fwd_fs=fwd_fs, fs=512, surfer_view=args.surfer_view, subject=test_config.subject_name)

# visu GT, output from 4dvarnet and phi(output) : check if the prior autoencoder is working / learned
if baselines: 
    plot_src_from_imgs({"GT":img_gt_morph, "MB": img_mb_morph, "phi(MB)": img_mb_ae_morph, "phi(GT)": img_mb_ae_gt_morph}, 
        ["GT","MB", "phi(MB)", "phi(GT)"])
    plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
    if args.show: 
        plt.show(block=False)
    else :
        plt.close()

    plot_src_from_imgs({"GT":img_gt_morph, "MB": img_mb_morph, "LSTM":img_lstm_morph, "1DCNN":img_cnn_morph}, ["GT","MB", "LSTM", "1DCNN"])
    plt.savefig(Path(figs_path, f"cortex_learning_based_idx_{idx}.png"))
    if args.show: 
        plt.show(MBock=False)
    else :
        plt.close()

    plot_src_from_imgs({"GT":img_gt_morph, "MB": img_mb_morph, "LSTM":img_lstm_morph, "1DCNN":img_cnn_morph, "MNE": img_mne_morph, "sLORETA":img_slo_morph}, 
                       ["GT","MB", "LSTM", "1DCNN", "MNE", "sLORETA"], 
                       subtitles=["\n0 -1"] + [f"\n{le:.3f} - {auc*100:.2f}" for le,auc in [(le_mb, auc_mb), (le_lstm, auc_lstm), (le_cnn, auc_cnn), (le_mne, auc_mne), (le_slo, auc_slo)]])
    plt.savefig(Path(figs_path, f"cortex_all_methods_{idx}.png"))
    if args.show: 
        plt.show(block=False)
    else :
        plt.close()

    plot_src_from_imgs({"MNE": img_mne_morph, "sLORETA":img_slo_morph}, ["MNE", "sLORETA"])
    plt.savefig(Path(figs_path, f"cortex_nl_based_idx_{idx}.png"))
    if args.show: 
        plt.show(block=False)
    else :
        plt.close()

else : 
    plot_src_from_imgs({"GT":img_gt_morph, "MB": img_mb_morph, "phi(MB)": img_mb_ae_morph, "phi(GT)": img_mb_ae_gt_morph}, ["GT","MB", "phi(MB)", "phi(GT)"])
    plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
    if args.show: 
        plt.show(block=False)
    else :
        plt.close()
    
    plot_src_from_imgs({"GT":img_gt_morph, "MB": img_mb_morph, "MNE": img_mne_morph, "sLORETA":img_slo_morph}, 
                       ["GT","MB", "MNE", "sLORETA"], 
                       subtitles=["\n0 -1"] + [f"\n{le:.3f} - {auc*100:.2f}" for le,auc in [(le_mb, auc_mb), (le_mne, auc_mne), (le_slo, auc_slo)]])
    plt.savefig(Path(figs_path, f"cortex_all_methods_{idx}.png"))
    if args.show: 
        plt.show(block=False)
    else :
        plt.close()



## Variational cost during the gradient descent (and obs and prior cost separately)
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(litmodel.solver.varc, '-', marker='o')
plt.xlabel('step')
plt.title("var cost")

plt.subplot(1,3,2)
plt.plot( litmodel.solver.obsc, '-', marker='o' )
plt.title("obs cost")
plt.xlabel('step')

plt.subplot(1,3,3)
plt.plot( litmodel.solver.pc , '-',  marker='o')
plt.title("prior cost")
plt.xlabel('step')
plt.savefig(Path(figs_path,f"var_cost_idx_{idx}.png"))
if args.show: 
    plt.show(MBock=False)
else: 
    plt.close() 

if baselines: 
    src_hat = {
        "GT": batch.tgt[idx_v,:,:].detach().numpy(),
        "mb": output[idx_v,:,:].detach().numpy(),
        "LSTM": lstm_output[idx_v,:,:].numpy(),
        "1DCNN": cnn_output[idx_v,:,:].numpy(),
        "MNE": mne_output,
        "sLORETA": slo_output
    }
else : 
    src_hat = {
        "GT": batch.tgt[idx_v,:,:].detach().numpy(),
        "mb": output[idx_v,:,:].detach().numpy(),
        "MNE": mne_output,
        "sLORETA": slo_output
    }



fig, axes = plt.subplots(figsize=(16, 10), nrows=1, ncols=len(list(src_hat.keys())))
i = 0
for m in src_hat.keys(): 
    tp = t_max / mne_info['sfreq']
    eeg_hat = fwd['sol']['data'] @ src_hat[m]
    eeg_hat = eeg_hat / np.abs(eeg_hat).max()
    reproj = mne.EvokedArray( data = eeg_hat, info = mne_info, tmin=-0.)
    reproj.plot_topomap(
        times=tp,
        colorbar=False, 
        axes = axes[i]
    )
    axes[i].set_title(f"{m}")
    i += 1
plt.savefig(Path(figs_path, f"eeg_reproj_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else:
    plt.close()



n_sources = fwd['sol']['data'].shape[1]
fig, axes = plt.subplots(figsize=(16, 10), nrows=2, ncols=len(list(src_hat.keys()))//2)
for i, m in enumerate(src_hat.keys()): 
    src_plot = src_hat[m].squeeze() / np.abs(src_hat[m]).max()
    for s in range( n_sources ):
        axes[i//axes.shape[1], i%axes.shape[1]].plot( src_plot[s,:] )
    axes[i//axes.shape[1], i%axes.shape[1]].set_title(f"{m}")
plt.savefig(Path(figs_path, f"wvfs_estim_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else:
    plt.close()

fig = plt.figure()
for i, (m,o) in enumerate(src_hat.items()): 
    normalised_output = o.squeeze()[:,t_max] / np.abs(o).squeeze()[:,t_max].max()
    ax = fig.add_subplot(1, len(list(src_hat.keys())), i+1, projection='3d')
    ax.scatter(fwd['source_rr'][:,0], fwd['source_rr'][:,1], fwd['source_rr'][:,2], c=normalised_output, marker='o')
    plt.title(f"{m}")

plt.show(block=False)

######### SAVE RESULTS 
if source_reordering_bsl: 
    saving_path =  Path(output_dir, "res", "intersubj_reord", test_config.simu_name)
else: 
    saving_path =  Path(output_dir, "res", "intersubj_reord", test_config.simu_name)
os.makedirs( saving_path , exist_ok=True)
for k,v in src_hat.items(): 
    ### --- SAVE SOURCES --- ##
    np.save( Path( saving_path, f"SOURCE_{k}_{idx}.npy" ), v.squeeze() )
    print(f"{v.squeeze().shape=}") 
    ### --- SAVE EEG --- ###
    if k == "GT": 
        np.save( Path( saving_path, f"EEG_{k}_{idx}.npy" ), eeg.squeeze() )
    else: 
        eeg_hat = leadfield @ v.squeeze()
        print(f"{eeg_hat.squeeze().shape=}") 
        np.save( Path( saving_path, f"EEG_{k}_{idx}.npy" ), eeg_hat.squeeze() )
### --- SAVE VAR COST --- ###
costs = np.zeros( (3, litmodel.solver.n_step) )
costs[0,:] = litmodel.solver.varc
costs[1,:] = litmodel.solver.obsc
costs[2,:] = litmodel.solver.pc
np.save( Path( saving_path, f"COSTS_{idx}.npy" ), costs )