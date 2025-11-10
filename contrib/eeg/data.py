### EEG
"""
For SEREEGA data: different way to create the dataset
- EsiDataset -> base dataset using TrainingItem as output for getitem
- EsiDatasetNoiseFixed -> Load EEG data simulated with a fixed noise (instead of adding noise when loading data)
- EsiDatasetNoiseSrc -> Add noise to source data (test with cosine similarity)
- EsiDatasetTw -> Load only a small time window of the data
- EsiDatasetAE -> Dataset to train autoencoder (output source+noise and source)
- EsiDatasetID -> Dataset to train direct inversion methods (do not use TrainingItem)

- EsiDatamodule -> pytorch lightning datamodule, builds the training/validation or test dataset, given the option among 
    all the different types of datasets.

    
*TODO* add NMM simulations dataset
*TODO* Better loading of the data (works but to complex)
"""
import functools as ft
import itertools
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import tqdm
from torch.utils.data import Dataset
from contrib.eeg import utils_eeg as utl


TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class EsiDataset(Dataset):
    """
    Dataset for ESI data of SEREEGA based simulations
    **inputs**
        - root_simu:str , root path of simulation data (ex: home/user/Data/simulation)
        - config_file: str , name of the configuration file of the simulation
        - simu_name: str , name of the simulation
        - subject_name: str, name of the subject
        - source_sampling: str, subsampling of the source space used
        - electrode_montage: str, name of the electrode montage used
        - to_load: int, number of samples to load from the dataset
        - snr_db: int, snr of the EEG data
        - noise_type={"white":1.}: dict, type of noise to add (dict with the color of the noise and a ratio of the total noise 
                                that corresponds to this component)
        - scaler_type="linear":str, type of scaling to use for normalisation (linear or max-max)
    """

    def __init__(
        self,
        datafolder,
        config_file,
        simu_name,
        subject_name,
        source_sampling,
        electrode_montage,
        to_load,
        snr_db,
        noise_type={"white":1.},
        scaler_type="linear",
        orientation="constrained",
        replace_root=False, 
        load_lf = True,
        subset_file = None, 
    ):
        super().__init__()
        #home = expanduser('~')
        self.datafolder = datafolder
        
        self.to_load = to_load
        self.load_lf = load_lf
        
        ## build path to simulation ##
        self.simu_path = Path(  
            self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "simu"
        )
        # print(f"--------------------{self.simu_path}")

        ### load confi ###
        # load config 
        with open(config_file) as f : 
            config_dict = json.load(f)

        self.config = {
            "subject_name": subject_name,
            "simu_name": simu_name, 
            "orientation": orientation, 
            "electrode_montage": electrode_montage, 
            "source_sampling": source_sampling, 
            "n_electrodes": config_dict['electrode_space']['n_electrodes'], 
            "n_sources": config_dict['source_space']['n_sources'] , 
            "fs": config_dict['rec_info']['fs'],
            "n_times": config_dict['rec_info']['n_times'], 
        }
        
        self.replace_root = replace_root
        self.snr = snr_db
        self.scaler_type = scaler_type

        info_file = Path( self.simu_path, simu_name ,f"{simu_name}{source_sampling}_match_json_file.json" )
        # load info to match the files
        with open(info_file) as f : 
            self.match_info_dict = json.load(f)
        if subset_file is None : 
            self.data_ids = np.array( 
                list( self.match_info_dict.keys() )
            )
        else : 
            with open(subset_file, 'r') as file:
                self.data_ids = np.array(
                    [line.rstrip() for line in file]
                )
        # samples to keep #
        n_samples = len(self.data_ids)
        if self.to_load > n_samples : 
            self.to_load = n_samples
            print(f"--- only {n_samples} available ---")

        if self.to_load != n_samples : 
            self.data_ids = self.data_ids[
                np.random.choice(np.arange(n_samples), self.to_load, replace=False)
            ]
        
        # metadata #
        self.md = [{}] * self.to_load #dict( zip(self.data_ids, {} ) )
        self.act_src = [[]] * self.to_load #dict( zip(self.data_ids, {} ) )
        
        ###### THIS PART IS TO CHANGE / SIMPLIFY #####
        for k in range(self.to_load) : 
            if self.replace_root : 
                md_json_file_name = self.replace_root_fn( 
                    self.match_info_dict[self.data_ids[k]]['md_json_file_name']
                )
                # with open( Path(self.datafolder, md_json_file_name) ) as f :
                with open( Path( md_json_file_name) ) as f : 
                    self.md[k] = json.load(f)
            else : 
                with open( Path(self.datafolder, self.match_info_dict[self.data_ids[k]]['md_json_file_name']) ) as f : 
                    self.md[k] = json.load(f)

            act_src = []
            for p in range(self.md[k]['n_patch']) : 
                act_src += self.md[k]['act_src'][f'patch_{p+1}']
            self.act_src[k] = act_src
        #self.act_src = torch.from_numpy( np.array(self.act_src) )

     
        self.max_eeg = torch.zeros([self.to_load, 1])
        self.max_src = torch.zeros([self.to_load, 1])
        self.alpha_lf = None
        
        if self.load_lf: # load leadfield
            from contrib.eeg.utils_eeg import load_mat
            model_path = Path(  
               self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "model"
        ) 
            self.leadfield = load_mat(Path(model_path, f"LF_{source_sampling}.mat"))['G']
        if self.scaler_type == "linear_bis":
            self.alpha_lf = 10**(utl.find_exp(self.leadfield.max()) + 1)

    def replace_root_fn(self, string): 
        """ 
        replace "root" of file name (handle differences in simulations)
        """
        simu_name = self.config['simu_name']
        split = string.split(simu_name)
        # print(f"_________{split=}")
        mod_string = Path( 
            self.simu_path, simu_name, *split[1].split('/') 
            )
        # print(f"_______{mod_string=}")
        return mod_string
    
    def __len__(self):
        return self.to_load

    def __getitem__(self, index):
        from contrib.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)
        root_simu = self.datafolder
        md = self.md[index]
        if self.replace_root : 
            eeg_file_name = self.replace_root_fn(self.match_info_dict[self.data_ids[index]]['eeg_file_name'] )
            act_src_file_name = self.replace_root_fn( self.match_info_dict[self.data_ids[index]]['act_src_file_name'] )
            # eeg = load_mat( Path( root_simu, eeg_file_name ) )['eeg_data']['EEG'] 
            # src = load_mat(Path( root_simu, act_src_file_name) )['Jact']['Jact']
            eeg = load_mat( Path( eeg_file_name ) )['eeg_data']['EEG'] 
            src = load_mat(Path( act_src_file_name) )['Jact']['Jact']
        else : 
            eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
            src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()

        
        # add noise to EEG data
        inf = np.min(eeg)
        sup = np.max(eeg)
        if self.snr < 50:
            eeg = array_range_scaling(
                add_noise_snr( self.snr,eeg ), inf, sup
            )       
        
        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, _ = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield, alpha_L=self.alpha_lf)
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        # self.alpha_lf = alpha_lf 

        return TrainingItem(input=torch.from_numpy(eeg).float(), tgt=torch.from_numpy(src_tot).float()) 

## DATASET FOR DIRECT INVERSION (do not return a TrainingItem object but just the EEG and source data)
class EsiDatasetID(EsiDataset): 
    def __init__(self, datafolder, config_file, simu_name, subject_name, source_sampling, electrode_montage, to_load, snr_db, noise_type={ "white": 1 }, scaler_type="linear", orientation="constrained", replace_root=False, load_lf=True, subset_file=None):
        super().__init__(datafolder, config_file, simu_name, subject_name, source_sampling, electrode_montage, to_load, snr_db, noise_type, scaler_type, orientation, replace_root, load_lf, subset_file)
    def __getitem__(self, index):
        batch= super().__getitem__(index)
        return batch.input, batch.tgt

##
class EsiDatasetNoiseFixed(Dataset):
    """
    LOAD DATA WITH A FIXED SNR (snr_db) IN ORDER TO HAVE REPRODUCIBLE EVALUATION RESULTS
    -> only change = file path for EEG data
    Dataset for ESI data of SEREEGA based simulations
    **inputs**
        - root_simu:str , root path of simulation data (ex: home/user/Data/simulation)
        - config_file: str , name of the configuration file of the simulation
        - simu_name: str , name of the simulation
        - subject_name: str, name of the subject
        - source_sampling: str, subsampling of the source space used
        - electrode_montage: str, name of the electrode montage used
        - to_load: int, number of samples to load from the dataset
        - snr_db: int, snr of the EEG data
        - noise_type={"white":1.}: dict, type of noise to add (dict with the color of the noise and a ratio of the total noise 
                                that corresponds to this component)
        - scaler_type="linear":str, type of scaling to use for normalisation (linear or max-max)
    """

    def __init__(
        self,
        datafolder,
        config_file,
        simu_name,
        subject_name,
        source_sampling,
        electrode_montage,
        to_load,
        snr_db,
        noise_type={"white":1.},
        scaler_type="linear",
        orientation="constrained",
        replace_root=False, 
        load_lf = True,
        subset_file = None
    ):
        super().__init__()
        #home = expanduser('~')
        self.datafolder = datafolder
        
        self.to_load = to_load
        self.load_lf = load_lf
        
        ## build path to simulation ##
        self.simu_path = Path(  
            self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "simu"
        )

        ### load confi ###
        # load config 
        with open(config_file) as f : 
            config_dict = json.load(f)

        self.config = {
            "subject_name": subject_name,
            "simu_name": simu_name, 
            "orientation": orientation, 
            "electrode_montage": electrode_montage, 
            "source_sampling": source_sampling, 
            "n_electrodes": config_dict['electrode_space']['n_electrodes'], 
            "n_sources": config_dict['source_space']['n_sources'] , 
            "fs": config_dict['rec_info']['fs'],
            "n_times": config_dict['rec_info']['n_times'], 
        }
        
        self.replace_root = replace_root
        self.snr = snr_db
        self.scaler_type = scaler_type

        info_file = Path( self.simu_path, simu_name ,f"{simu_name}{source_sampling}_match_json_file.json" )
        # load info to match the files
        with open(info_file) as f : 
            self.match_info_dict = json.load(f)
        if subset_file is None : 
            self.data_ids = np.array( 
                list( self.match_info_dict.keys() )
            )
        else : 
            with open(subset_file, 'r') as file:
                self.data_ids = np.array(
                    [line.rstrip() for line in file]
                )
        ## change match_info_dict pathes...
        for i in range(len(self.data_ids)) :
            self.match_info_dict[self.data_ids[i]]['eeg_file_name'] = Path(
                datafolder, subject_name, orientation, electrode_montage, source_sampling, "simu", simu_name, "eeg", f"{self.snr}db", f"{self.data_ids[i].split('_')[-1]}_eeg.mat"
                )
            self.match_info_dict[self.data_ids[i]]['act_src_file_name'] = Path(
                datafolder, subject_name, orientation, electrode_montage, source_sampling, "simu", simu_name, "sources", "Jact", f"{self.data_ids[i].split('_')[-1]}_src_act.mat"
                )
            self.match_info_dict[self.data_ids[i]]['md_json_file_name'] = Path(
                datafolder, subject_name, orientation, electrode_montage, source_sampling, "simu", simu_name, "md", f"{self.data_ids[i].split('_')[-1]}_md_json_flie.json"
                )
            # noise_src_file_name 
            
        # samples to keep #
        n_samples = len(self.data_ids)
        if self.to_load > n_samples : 
            self.to_load = n_samples
            print(f"--- only {n_samples} available ---")

        if self.to_load != n_samples : 
            self.data_ids = self.data_ids[
                np.random.choice(np.arange(n_samples), self.to_load, replace=False)
            ]
        
        # metadata #
        self.md = [{}] * self.to_load #dict( zip(self.data_ids, {} ) )
        self.act_src = [[]] * self.to_load #dict( zip(self.data_ids, {} ) )
        for k in range(self.to_load) : 
            with open( Path(self.datafolder, self.match_info_dict[self.data_ids[k]]['md_json_file_name']) ) as f : 
                self.md[k] = json.load(f)

            act_src = []
            for p in range(self.md[k]['n_patch']) : 
                act_src += self.md[k]['act_src'][f'patch_{p+1}']
            self.act_src[k] = act_src
        #self.act_src = torch.from_numpy( np.array(self.act_src) )

     
        self.max_eeg = torch.zeros([self.to_load, 1])
        self.max_src = torch.zeros([self.to_load, 1])
        self.alpha_lf = None

        if self.load_lf: 
            from contrib.eeg.utils_eeg import load_mat
            model_path = Path(  
               self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "model"
        ) 
            self.leadfield = load_mat(Path(model_path, f"LF_{source_sampling}.mat"))['G']
        if self.scaler_type == "linear_bis":
            self.alpha_lf = 10**(utl.find_exp(self.leadfield.max()) + 1)

    def __len__(self):
        return self.to_load

    def __getitem__(self, index):
        from contrib.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)
        root_simu = self.datafolder
        # md = self.md[index]
        eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
        src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()

        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, alpha_lf = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield, alpha_L=self.alpha_lf )
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        # self.alpha_lf = alpha_lf 

        return TrainingItem(input=torch.from_numpy(eeg).float(), tgt=torch.from_numpy(src_tot).float()) 

import pytorch_lightning as pl

class EsiDatamodule(pl.LightningDataModule):
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2, config_file=None, subset_name=None, time_window=False, noise_fixed=False, di=False, src_noise=False):
        super().__init__()
        self.dl_kw = dl_kw
        self.time_window = time_window
        self.per_valid = per_valid
        self.train_ds = None
        self.val_ds = None
        self.noise_fixed = noise_fixed
        self.di = di # direct inverstion
        self.src_noise = src_noise

        self.dataset_kw =  dataset_kw
        if config_file is None : 
            config_file = Path(
                dataset_kw['datafolder'], 
                dataset_kw['subject_name'], dataset_kw['orientation'], dataset_kw['electrode_montage'], dataset_kw['source_sampling'], "simu",
                dataset_kw['simu_name'], f"{dataset_kw['simu_name']}{dataset_kw['source_sampling']}_config.json"
            )
        self.dataset_kw.update({'config_file': config_file})

        if subset_name.lower() != 'none': 
            subset_file = Path(
                dataset_kw['datafolder'], 
                dataset_kw['subject_name'], dataset_kw['orientation'], dataset_kw['electrode_montage'], dataset_kw['source_sampling'], "simu",
                dataset_kw['simu_name'], f"{subset_name}.txt"
            )
            self.dataset_kw.update({'subset_file': subset_file})
        else : 
            self.dataset_kw.update({'subset_file': None})

    def setup(self, stage):
        if stage == "test": 
            if self.time_window: 
                self.test_ds = TWEsiDataset(
                    **self.dataset_kw
                )
            elif self.noise_fixed: 
                self.test_ds = EsiDatasetNoiseFixed(
                    **self.dataset_kw
                )
            elif self.di: 
                self.test_ds = EsiDatasetID(
                    **self.dataset_kw
                )
            elif self.src_noise: 
                self.test_ds = EsiDatasetNoiseSrc(
                    **self.dataset_kw
                )
            else : 
                self.test_ds = EsiDataset(
                    **self.dataset_kw
                )
        else : 
            if self.time_window: 
                ds_dataset = TWEsiDataset(
                    **self.dataset_kw
                )
            elif self.noise_fixed: 
                ds_dataset = EsiDatasetNoiseFixed(
                    **self.dataset_kw
                )
            elif self.di:  # direct inversion
                ds_dataset = EsiDatasetID(
                    **self.dataset_kw
                )
            elif self.src_noise: 
                ds_dataset = EsiDatasetNoiseSrc(
                    **self.dataset_kw
                )
            else :
                ds_dataset = EsiDataset(
                    **self.dataset_kw
                )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            )
        

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

import sys


def scaled_data(y, x, scaling_type="linear_bis", leadfield=None, alpha_L=1e3): 
    """
    Inputs :
        x : EEG data - numpy array - shape (Ne,T)
        y : source data - numpy array - shape (Ns,T)
        scaling_type : type of scaling -> None (raw data), linear, nlinear or linear bis
        leadfield : leadfield matrix (required for linear_bis scaling)

    Outputs : 
        scaled data + scaling factor(s) in the order :
        scaled_y, scaled_x, scaled_lf, y_scale_factor, x_scale_factor, leadfield_scale_factor
    """
    if scaling_type.lower()=='raw' : 
        return y, x, leadfield, 1, 1, 1
    elif scaling_type.lower()=="linear": 
        max_y = np.max(np.abs(y))
        return y / max_y, x / max_y, leadfield, max_y, max_y, 1
    elif scaling_type.lower()=="nlinear":
        max_y = np.max(np.abs(y))
        max_x = np.max(np.abs(x))
        return y / max_y, x / max_x, leadfield, max_y, max_x, 1
    elif scaling_type.lower()=="linear_bis": 
        if leadfield is None : 
            sys.exit(f"leadfield required for scaling {scaling_type}")
        # alpha_L = 1e3
        max_y = np.max(np.abs(y))   
        # return y / max_y, x / (1e-3*max_y), leadfield / alpha_L, max_y, 1e-3*max_y, alpha_L
        return y / max_y, (alpha_L *x) / (max_y), leadfield / alpha_L, max_y, max_y / alpha_L, alpha_L
    elif scaling_type.lower() =="leadfield": 
        if leadfield is None : 
            sys.exit(f"leadfield required for scaling {scaling_type}")
        alpha_L = 10*np.max(leadfield)
        max_y = np.max(np.abs(y)) 
        
        return y / max_y, alpha_L*x / max_y, leadfield / alpha_L, max_y, alpha_L/max_y, alpha_L

    else : 
        sys.exit(f"unsupported scaling type {scaling_type}")


############## time window dataset
class TWEsiDataset(EsiDataset): 
    def __init__(self, win_length = 16, **kwargs):
        super().__init__(**kwargs)
        self.win_length = win_length
        self.win_centers = [None] * self.to_load

    def __getitem__(self, index):  
        from contrib.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)     
        from contrib.eeg.data import scaled_data
        root_simu = self.datafolder
        md = self.md[index]
        if self.replace_root : 
            eeg_file_name = self.replace_root_fn(self.match_info_dict[self.data_ids[index]]['eeg_file_name'] )
            act_src_file_name = self.replace_root_fn( self.match_info_dict[self.data_ids[index]]['act_src_file_name'] )
            eeg = load_mat( Path( root_simu, eeg_file_name ) )['eeg_data']['EEG'] 
            src = load_mat(Path( root_simu, act_src_file_name) )['Jact']['Jact']
        else : 
            eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
            src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()
        
        if self.win_centers[index] is None :
            src_tot = torch.from_numpy(src_tot).float()
            # t_max = torch.argmax( src_tot.sum(axis=0).abs() )
            t_act = torch.where( src_tot.sum(axis=0).abs() > 0.1*src_tot.sum(axis=0).abs().max() )[0]

            window_center = t_act[torch.randint(0,len(t_act),(1,1)).item()]
            n_times = eeg.shape[1]
            ## check if window is not out of bounds
            if window_center - self.win_length//2 < 0: 
                window_center = self.win_length//2
            if window_center + self.win_length//2 >= n_times:
                window_center = n_times - self.win_length//2
            self.win_centers[index] = window_center
        else : 
            window_center = self.win_centers[index]

        eeg = eeg[:, window_center-self.win_length//2 : window_center+self.win_length//2]
        src_tot = src_tot[:, window_center-self.win_length//2 : window_center+self.win_length//2]
        # print(f"window center : {window_center}")
        # print(f"window range: {window_center-self.win_length//2},{window_center+self.win_length//2}")
        # add noise to EEG data
        inf = np.min(eeg)
        sup = np.max(eeg)
        if self.snr < 50:
            eeg = array_range_scaling(
                add_noise_snr( self.snr,eeg ), inf, sup
            )       
        
        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, alpha_lf = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield , alpha_L=1e3)
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        self.alpha_lf = alpha_lf 
        # print(eeg.shape, src_tot.shape)
        if not torch.is_tensor(eeg): 
            eeg = torch.from_numpy(eeg).float()
        if not torch.is_tensor(src_tot):
            src_tot = torch.from_numpy(src_tot).float()
        return TrainingItem(
            input=eeg, 
            tgt=src_tot )


######################################
## Test to add source noise to enhance results with COSINE similarity
class EsiDatasetNoiseSrc(Dataset):
    """
    Dataset for ESI data of SEREEGA based simulations - source noise for better cosine similarity.
    **inputs**
        - root_simu:str , root path of simulation data (ex: home/user/Data/simulation)
        - config_file: str , name of the configuration file of the simulation
        - simu_name: str , name of the simulation
        - subject_name: str, name of the subject
        - source_sampling: str, subsampling of the source space used
        - electrode_montage: str, name of the electrode montage used
        - to_load: int, number of samples to load from the dataset
        - snr_db: int, snr of the EEG data
        - noise_type={"white":1.}: dict, type of noise to add (dict with the color of the noise and a ratio of the total noise 
                                that corresponds to this component)
        - scaler_type="linear":str, type of scaling to use for normalisation (linear or max-max)
    """

    def __init__(
        self,
        datafolder,
        config_file,
        simu_name,
        subject_name,
        source_sampling,
        electrode_montage,
        to_load,
        snr_db,
        noise_type={"white":1.},
        scaler_type="linear",
        orientation="constrained",
        replace_root=False, 
        load_lf = True,
        subset_file = None, 
    ):
        super().__init__()
        #home = expanduser('~')
        self.datafolder = datafolder
        
        self.to_load = to_load
        self.load_lf = load_lf
        
        ## build path to simulation ##
        self.simu_path = Path(  
            self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "simu"
        )
        # print(f"--------------------{self.simu_path}")

        ### load confi ###
        # load config 
        with open(config_file) as f : 
            config_dict = json.load(f)

        self.config = {
            "subject_name": subject_name,
            "simu_name": simu_name, 
            "orientation": orientation, 
            "electrode_montage": electrode_montage, 
            "source_sampling": source_sampling, 
            "n_electrodes": config_dict['electrode_space']['n_electrodes'], 
            "n_sources": config_dict['source_space']['n_sources'] , 
            "fs": config_dict['rec_info']['fs'],
            "n_times": config_dict['rec_info']['n_times'], 
        }
        
        self.replace_root = replace_root
        self.snr = snr_db
        self.src_snr = 35
        self.scaler_type = scaler_type

        info_file = Path( self.simu_path, simu_name ,f"{simu_name}{source_sampling}_match_json_file.json" )
        # load info to match the files
        with open(info_file) as f : 
            self.match_info_dict = json.load(f)
        if subset_file is None : 
            self.data_ids = np.array( 
                list( self.match_info_dict.keys() )
            )
        else : 
            with open(subset_file, 'r') as file:
                self.data_ids = np.array(
                    [line.rstrip() for line in file]
                )
        # samples to keep #
        n_samples = len(self.data_ids)
        if self.to_load > n_samples : 
            self.to_load = n_samples
            print(f"--- only {n_samples} available ---")

        if self.to_load != n_samples : 
            self.data_ids = self.data_ids[
                np.random.choice(np.arange(n_samples), self.to_load, replace=False)
            ]
        
        # metadata #
        self.md = [{}] * self.to_load #dict( zip(self.data_ids, {} ) )
        self.act_src = [[]] * self.to_load #dict( zip(self.data_ids, {} ) )
        
        ###### THIS PART IS TO CHANGE / SIMPLIFY #####
        for k in range(self.to_load) : 
            if self.replace_root : 
                md_json_file_name = self.replace_root_fn( 
                    self.match_info_dict[self.data_ids[k]]['md_json_file_name']
                )
                # with open( Path(self.datafolder, md_json_file_name) ) as f :
                with open( Path( md_json_file_name) ) as f : 
                    self.md[k] = json.load(f)
            else : 
                with open( Path(self.datafolder, self.match_info_dict[self.data_ids[k]]['md_json_file_name']) ) as f : 
                    self.md[k] = json.load(f)

            act_src = []
            for p in range(self.md[k]['n_patch']) : 
                act_src += self.md[k]['act_src'][f'patch_{p+1}']
            self.act_src[k] = act_src
        #self.act_src = torch.from_numpy( np.array(self.act_src) )

     
        self.max_eeg = torch.zeros([self.to_load, 1])
        self.max_src = torch.zeros([self.to_load, 1])
        self.alpha_lf = None
        
        if self.load_lf: # load leadfield
            from contrib.eeg.utils_eeg import load_mat
            model_path = Path(  
               self.datafolder, subject_name, orientation, electrode_montage ,source_sampling, "model"
        ) 
            self.leadfield = load_mat(Path(model_path, f"LF_{source_sampling}.mat"))['G']
        if self.scaler_type == "linear_bis":
            self.alpha_lf = 10**(utl.find_exp(self.leadfield.max()) + 1)

    def replace_root_fn(self, string): 
        """ 
        replace "root" of file name (handle differences in simulations)
        """
        simu_name = self.config['simu_name']
        split = string.split(simu_name)
        # print(f"_________{split=}")
        mod_string = Path( 
            self.simu_path, simu_name, *split[1].split('/') 
            )
        # print(f"_______{mod_string=}")
        return mod_string
    
    def __len__(self):
        return self.to_load

    def __getitem__(self, index):
        from contrib.eeg.utils_eeg import (add_noise_snr, array_range_scaling,
                                           load_mat)
        root_simu = self.datafolder
        md = self.md[index]
        if self.replace_root : 
            eeg_file_name = self.replace_root_fn(self.match_info_dict[self.data_ids[index]]['eeg_file_name'] )
            act_src_file_name = self.replace_root_fn( self.match_info_dict[self.data_ids[index]]['act_src_file_name'] )
            # eeg = load_mat( Path( root_simu, eeg_file_name ) )['eeg_data']['EEG'] 
            # src = load_mat(Path( root_simu, act_src_file_name) )['Jact']['Jact']
            #eeg = load_mat( Path( eeg_file_name ) )['eeg_data']['EEG'] 
            src = load_mat(Path( act_src_file_name) )['Jact']['Jact']
        else : 
            #eeg = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['eeg_file_name'] ) )['eeg_data']['EEG'] 
            src = load_mat( Path( root_simu, self.match_info_dict[self.data_ids[index]]['act_src_file_name']) )['Jact']['Jact']

        #reconstruct source data
        src_tot = np.zeros([self.config['n_sources'], self.config['n_times']])
        src_tot[self.act_src[index], :] = src.copy()

        ## add noise to source data ##
        inf = np.min(src_tot)
        sup = np.max(src_tot)
        src_tot = array_range_scaling(
                add_noise_snr( self.src_snr,src_tot ), inf, sup
        )    
        eeg = self.leadfield @ src_tot   
        
        # add noise to EEG data
        inf = np.min(eeg)
        sup = np.max(eeg)
        if self.snr < 50:
            eeg = array_range_scaling(
                add_noise_snr( self.snr,eeg ), inf, sup
            )       
        
        # scale data
        eeg, src_tot, _, alpha_eeg, alpha_src, _ = scaled_data( 
            eeg, src_tot, scaling_type=self.scaler_type, leadfield=self.leadfield, alpha_L=self.alpha_lf)
        self.max_eeg[index] = alpha_eeg
        self.max_src[index] = alpha_src
        # self.alpha_lf = alpha_lf 

        return TrainingItem(input=torch.from_numpy(eeg).float(), tgt=torch.from_numpy(src_tot).float()) 

############# dataset for autoencoder #########
class EsiDatasetAE(EsiDataset):
    def __init__(self, datafolder, config_file, simu_name, subject_name, source_sampling, electrode_montage, to_load, snr_db, noise_type={ "white": 1 }, scaler_type="linear", orientation="constrained", replace_root=False, load_lf=True, subset_file=None, add_noise=True):
        super().__init__(datafolder, config_file, simu_name, subject_name, source_sampling, electrode_montage, to_load, snr_db, noise_type, scaler_type, orientation, replace_root, load_lf, subset_file)
        self.add_noise = add_noise

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.add_noise:
            # add noise to data - random snr in a range
            noise = torch.randn_like(data.tgt) 
            snr_db_range = np.arange(-15, 0, 1)
            snr_db = np.random.choice(snr_db_range)
            snr = 10**(snr_db/10)

            alpha_snr = (1/np.sqrt(snr))*(data.tgt.norm() / noise.norm())
            data_noisy = data.tgt + alpha_snr*noise
            return data_noisy, data.tgt
        else : 
            return data.tgt, data.tgt 
        
class EsiDatamoduleAE(EsiDatamodule):
    def __init__(self, dataset_kw, dl_kw, per_valid=0.2, config_file=None, subset_name=None):
        super().__init__(dataset_kw, dl_kw, per_valid, config_file, subset_name)

    def setup(self, stage) : 
        if stage == "test": 
            self.test_ds = EsiDatasetAE(
                **self.dataset_kw
            )
        else : 
            ds_dataset = EsiDatasetAE(
                **self.dataset_kw
            )
            self.dataset_kw['to_load'] = len(ds_dataset) #ds_dataset.to_load
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                ds_dataset,
                [int(self.dataset_kw['to_load'] * (1 - self.per_valid)), 
                 int(self.dataset_kw['to_load']) - int(self.dataset_kw['to_load']*(1 - self.per_valid))],
            )