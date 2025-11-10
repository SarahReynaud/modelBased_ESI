import sys

# import colorednoise as cn
import mne
import numpy as np
from mne.datasets import sample
from mne.io import read_raw_fif
from scipy.io import loadmat, matlab
from torch import nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import json
from pathlib import Path

from math import log10, floor
import os


def build_path(folders): 
    from os.path import join
    return join(*folders)

def load_mat(filename):
    """
    (from stackoverflow)
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

def add_noise_snr(snr_db, signal, noise_type = {"white":1.}, return_noise=False): 
    """  
    Return a signal which is a linear combination of signal and noise with a ponderation to have a given snr
    noise_type : dict key = type of noise, value = ponderation of the noise (for example use white and pink noise)
    """

    snr     = 10**(snr_db/10)
    noise = 0
    dims = [signal.shape[i] for i in range(len(signal.shape))]
    for n_type, pond in noise_type.items() : 
        if n_type=="white": 
            noise = noise + pond * np.random.randn( *dims )
        if n_type=="pink" : 
            beta = (1.8-0.2)*np.random.rand(1) + 0.2
            ## removed for experiments on jeanzay ##
            # noise = noise + pond * cn.powerlaw_psd_gaussian(beta, signal.shape)
            noise = None
            sys.exit("not implemented")

    if len(signal.shape)==2: 
        # 2D signal
        x = signal + (noise/np.linalg.norm(noise))*(np.linalg.norm(signal)/np.sqrt(snr))
    elif len(signal.shape)==3:
        # batch data 
        noise_norm = np.expand_dims( np.linalg.norm(noise, axis=(1,2)), (1,2) )
        sig_norm = np.expand_dims( np.linalg.norm(signal, axis=(1,2)), (1,2) )
        x = signal + (noise/noise_norm)*(sig_norm/np.sqrt(snr))
        #x = signal + (noise/np.linalg.norm(noise, axis=(1,2)))*(np.linalg.norm(signal,axis=(1,2))/np.sqrt(snr))
    else: 
        x = None
        sys.exit('Signal must be of dimension 2 (unbatched data) or 3 (batched data)')
        
    if return_noise : 
        return x, noise
    else: 
        return x

def tensor_range_scaling(x, inf, sup): 
    """
    rescale x so that it max(x) == sup and min(x) == inf.
    !! works for x a tensor of a single tensor (not a batch)
    returns rescaled_x, the rescaled tensor
    """
    x_maxs = torch.max(x)
    x_mins = torch.min(x)

    scale_factor = (sup - inf) / (x_maxs - x_mins) 
    rescaled_x = (x - x_mins) * scale_factor + x_mins
    return rescaled_x

def array_range_scaling(x, inf, sup): 
    """
    rescale x so that it max(x) == sup and min(x) == inf.
    !! works for x a tensor of a single tensor (not a batch)
    returns rescaled_x, the rescaled tensor
    """
    x_maxs = np.max(x)
    x_mins = np.min(x)

    scale_factor = (sup - inf) / (x_maxs - x_mins) 
    rescaled_x = (x - x_mins) * scale_factor + x_mins
    return rescaled_x

def get_neighbors(tris, verts): 
    n_verts = len(verts[0]) + len(verts[1])

    neighbors = [list() for _ in range(n_verts)]
    i = 0
    for hem in range(2): 
        
        idx_tris_old = np.sort(np.unique(tris[hem])).astype(np.int64)
        idx_vert_old = np.sort(np.unique(verts[hem])).astype(np.int64)

        missing_verts = np.setdiff1d(idx_tris_old, idx_vert_old)
        #idx_tris_new = np.arange(0, len(idx_tris_old))
        idx_vert_new = np.arange(0, len(idx_vert_old))

        vertices_lin = np.zeros((idx_vert_old.max()+1,1))
        vertices_lin[idx_vert_old,0] = i + idx_vert_new
        vertices_lin = vertices_lin.astype(np.int64)
        
        for v in verts[hem]: 
            triangles_of_v = np.squeeze(tris[hem] == v)
            triangles_of_v = np.squeeze(tris[hem][np.sum(triangles_of_v, axis=1) > 0])

            neighbors_of_v = np.unique(triangles_of_v)
            neighbors_of_v = neighbors_of_v[neighbors_of_v != v]
            neighbors_of_v = np.setdiff1d(neighbors_of_v, missing_verts)   
            

            #print(f"vert : {v}, {len(vertices_lin[neighbors_of_v,0])}")
            neighbors[i] = list( vertices_lin[neighbors_of_v,0] )
            i += 1

    l_max           = np.amax( np.array([len(l) for l in neighbors]) )
    neighb_array    = np.zeros( (len(neighbors), l_max) )
    for i in range(len(neighbors) ) : 
        l = neighbors[i]
        neighb_array[i,:len(l)] = l
        if len(l)<l_max: 
            neighb_array[i,len(l):] = None 
    neighb_array = np.nan_to_num(neighb_array, nan=-1)
    return neighb_array.astype(int)

# patch 
def get_patch(order, idx, neighbors): 
    new_idx = np.array( [idx], dtype=np.int64 )
    #print(new_idx)

    if order == 0: 
        return new_idx
    
    else: 
        # for each order, find roder one neighbors of the current sources in patch
        for _ in range(order): 
            neighb = np.unique( neighbors[new_idx,:] )
            #neighb = neighb[~np.isnan(neighb)].astype(np.int64)
            neighb = neighb[neighb>0].astype(np.int64)
            #neighb = np.array(neighb, dtype=np.int64)

            #print(f"neighbors: {neighb}")
            new_idx = np.append( new_idx, neighb )
            #print(f"new indices: {new_idx}")
            
        
        return np.unique(new_idx)
    

### simu related 
def get_component_extended_src(order, seed, neighbors, spos, erp_params, erp_dev, timeline):

    amplitude = erp_params['ampl']
    patch = get_patch(order, seed, neighbors)
    n_source_in_patch = patch.shape[0]

    c = []
    if order > 0 : 
        seed_pos = spos[seed,:]
        d_in_patch = np.sqrt(
            np.sum(
            (seed_pos - spos[patch,:])**2,1
            )
        )
        #print(f"size patch {patch.shape}, dist size : {d_in_patch.shape}")
        patch_dim = np.max(d_in_patch)
        #d_in_patch = d_in_patch/patch_dim
        #print(f"patch_dim {patch_dim}")
        #sig = patch_dim / np.sqrt(2*np.log10(2)) #/2
        sig = np.max(d_in_patch)/2
        #print(f"sigma dist : {sig}")
        ampl = amplitude * np.exp(-0.5*(d_in_patch/sig)**2)
        
        for s in range(n_source_in_patch):

            tmp_c = erp_component(
                source=patch[s], 
                erp_params= {'ampl':ampl[s], 'width':erp_params['width'], 'center':erp_params['center']}, 
                erp_dev=erp_dev, 
                timeline=timeline)

            c.append(tmp_c)

    else : 
        c.append( erp_component(patch[0], erp_params, erp_dev, timeline) )
    return c, patch, patch_dim


class erp():
    def __init__(self, source, params, timeline) -> None:
        self.source = source
        self.params = params
        self.timeline = timeline
    
    def signal(self) :
        t_vec = np.arange(
            0, self.timeline['length']*1e-3, 1/self.timeline['srate']
        )
        #t_vec = np.linspace(
        #    0,
        #    (self.timeline['length']/1e3*self.timeline['srate'] - 1)/self.timeline['srate'],
        #     int(self.timeline['length']/1e3*self.timeline['srate']) )
        #print(t_vec.shape)
        # self.time = t_vec
        #print(f"ampl : {self.params['ampl']}, center : {self.params['center']}, width : {self.params['width']}")
        center = self.params['center']*1e-3
        sgm = self.params['width']*1e-3 / 6
        signl = self.params['ampl'] * np.exp(-0.5*((t_vec - center)/sgm)**2)

        return t_vec, signl  
    
def erp_component( source, erp_params, erp_dev, timeline ):
    # todo : take erp_dev into account, for now we will do as if deviation parameters were
    # taken into account elsewhere in the code
    return erp(source=source, params=erp_params, timeline=timeline)


def generate_scalp_data(c_tot, leadfield, timeline):
    n_times = int(timeline['length']*1e-3*timeline['srate'])
    #scalp_data = np.zeros((leadfield.shape[0], timeline['length']*1e-3*timeline['srate']))
    act_src_idx = [c.source for c in c_tot]
    act_leadfield = leadfield[:, act_src_idx]
    act_src = np.zeros((len(act_src_idx), n_times))
    for i in range(len(c_tot)) : 
        _, act_src[i,:] = c_tot[i].signal()

    scalp_data = act_leadfield @ act_src
    return scalp_data, act_src

def make_sample_montage():
    data_path = sample.data_path() 
    fname_raw = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = read_raw_fif(fname_raw) 
    raw.pick_types(meg=False, eeg=True, stim=False, exclude=()).load_data()
    raw.pick_types(eeg=True)

    dig_montage = raw.info.get_montage()
    return dig_montage

  
##------------ build source positions--------------------- ##

def simu_data_visu( fwd_obj, n_times, fs, eeg_snr,  plot=True):
    leadfield = fwd_obj['sol']['data']
    spos = fwd_obj['source_rr']*1e3

    x_min = spos[:, 0].min()
    x_max = spos[:, 0].max()
    x_cond = np.squeeze(np.where(spos[:, 0] < (x_min + (x_max - x_min) / 3)))
    z_min = spos[:, 2].min()
    z_max = spos[:, 2].max()
    z_cond = np.setdiff1d(
        np.arange(leadfield.shape[1]),
        np.squeeze(np.where(spos[:, 2] > (z_min + 3 * (z_max - z_min) / 4))),
    )
    z_cond = np.setdiff1d(
        z_cond,
        np.squeeze(np.where(spos[:, 2] < (z_min + (z_max - z_min) / 4))),
    )
    y_min = spos[:, 1].min()
    y_max = spos[:, 1].max()
    y_cond = np.setdiff1d(
        np.arange(leadfield.shape[1]),
        np.squeeze(np.where(spos[:, 1] > (y_min + 3 * (y_max - y_min) / 4))),
    )
    y_cond = np.setdiff1d(
        y_cond,
        np.squeeze(np.where(spos[:, 1] < (y_min + (y_max - y_min) / 4))),
    )
    cond = np.array(list(set(x_cond).intersection(z_cond).intersection(y_cond)))

    seed_1 = cond[np.argmin(spos[cond, 1])]
    seed_2 = cond[np.argmax(spos[cond, 1])]
    seeds = [seed_2]#, seed_2]
    print(f"{seed_1=}, {seed_2=}")
    src = np.zeros((leadfield.shape[1], n_times))
    print(f"{src.shape=}") ###

    neighbors = get_neighbors(
        [fwd_obj["src"][0]["use_tris"], fwd_obj["src"][1]["use_tris"]],
        [fwd_obj["src"][0]["vertno"], fwd_obj["src"][1]["vertno"]],
        )
    
    extension_order = 2
    patch_1 = get_patch(extension_order, seed_1, neighbors)
    patch_2 = get_patch(extension_order, seed_2, neighbors)

    if plot : 
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(*spos.transpose(), s=10)
        ax.scatter(spos[patch_1, 0], spos[patch_1, 1], spos[patch_1, 2], color="red", s=50)
        ax.scatter(spos[patch_2, 0], spos[patch_2, 1], spos[patch_2, 2], color="red", s=50)
        plt.title("active sources")
        plt.show(block=False)

    # get the components of the patch
    duree = np.floor( 1000*(n_times/fs) )
    base_amplitude = 10 / 20.6  # 20.6 = mean number of sources per region
    p_amplitude_dev = 0.5
    base_width = 50
    p_width_dev = 0.02
    base_center = np.floor( duree / 2)
    p_center_dev = 0.8

    p_range_ampl = np.array(
        [
            base_amplitude - p_amplitude_dev * base_amplitude,
            base_amplitude + p_amplitude_dev * base_amplitude,
        ]
    )
    p_range_width = np.array(
        [base_width - p_width_dev * base_width, base_width + p_width_dev * base_width]
    )
    p_range_center = np.array(
        [base_center - p_center_dev * base_center, base_center + p_center_dev * base_center]
    )

    erp_dev_intra_patch = {"ampl": 0, "width": 0, "center": 0}
    timeline = {"n": 1, "srate": fs, "length": duree, "marker": "event1", "prestim": 0}
    c_tot = []
    patches_sereega = []
    for seed in seeds:
        erp_params = {
            "ampl": np.random.uniform(p_range_ampl[0], p_range_ampl[1], 1),
            "width": np.ceil(np.random.uniform(p_range_width[0], p_range_width[1], 1)),
            "center": np.ceil(np.random.uniform(p_range_center[0], p_range_center[1], 1)),
        }
        print(erp_params["center"])
        [c, patch, patch_dim] = get_component_extended_src(
            extension_order,
            seed,
            neighbors,
            spos,
            erp_params,
            erp_dev_intra_patch,
            timeline,
        )
        c_tot += c
        patches_sereega.append(patch)

    [eeg, src_sigs] = generate_scalp_data(c_tot, leadfield, timeline)
    print(f"{eeg.shape=}, {src.shape=}")

    eeg = add_noise_snr(
        snr_db=eeg_snr, signal=eeg, noise_type={"white": 1.0}
    )
    #eeg = eeg.numpy()
    eeg = array_range_scaling(
        eeg, eeg.min(), eeg.max()
    )

    act_src_sereega = [s for p in patches_sereega for s in p]
    src[act_src_sereega, :] = src_sigs
    
    
    return eeg, src, seeds


# Global Field Power (GFP) scaling
def gfp_scaling(y, x, L): 
    # y: ground truth EEG data
    # x: source distibution, unscaled
    # L: leadfield matrix

    x_scaled = torch.zeros_like(x)
    y_pred = torch.matmul( L, x ) 

    for t in range(x.shape[1]): #time instant by time instant
        if torch.std(y_pred[:,t]) == 0: 
            denom = 1
        else : 
            denom = torch.std(y_pred[:,t])
        x_scaled[:,t] = x[:,t] * ( torch.std(y[:,t]) / denom ) #torch.std(M_pred[:,t]) )
    
    return x_scaled
def gfp_scale(y_hat, m_gt, leadfield, dim=1 ): 
    """
    inputs : 
        y_hat : estimated source data
        m_gt : ground truth EEG data
        leadfield : leadfield matriw
        dim : dimension along which to compute the standard deviations : should be the spatial dimension of the EEG data.
    """
    dev = leadfield.device

    m_hat = torch.matmul(leadfield, y_hat.detach().to(dev))
    std_hat = m_hat.std(dim=dim)
    std_hat[m_hat.std(dim=dim)==0.] = 1 # to take care of division by zero
    std_hat = std_hat.to(dev)
    gt_std = m_gt.std(dim=dim).to(dev)
    alpha = (gt_std/std_hat)
    
    return y_hat.to(dev) * alpha.unsqueeze(1).to(dev)

###--------------------------------------------------EVAL--------------------------------------------###
###---------------------------------------FOR HYDRA--------------------------------------------###
def load_fwd( datafolder:str, head_model_dict:dict, fwd_name:str, scaler_type:str=None, src_mapping=None ):
    from os.path import expanduser, join
    from mne import convert_forward_solution, read_forward_solution
    # home = expanduser('~')
    model_path = join( 
        datafolder,
        head_model_dict['subject_name'], head_model_dict['orientation'], head_model_dict['electrode_montage'], head_model_dict['source_sampling'], 
        "model" )
    fwd =  read_forward_solution( 
        join( model_path, fwd_name )
    )
    if head_model_dict['orientation'] == 'constrained' : 
        fwd = convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
        )
    if scaler_type == "linear_bis": 
        print("NORMALIZATION LEADFIELD")
        alpha_lf = 10**(find_exp(np.abs(fwd['sol']['data']).max()) + 1)
        fwd['sol']['data'] = fwd['sol']['data'] / alpha_lf
    elif scaler_type == "leadfield": 
        print("NORMALIZATION LEADFIELD")
        fwd['sol']['data'] = fwd['sol']['data'] / (10*np.max(fwd['sol']['data']))
    else : 
        pass

    if src_mapping is not None :
        fwd = fwd[:,src_mapping]
    
    return fwd

### when using fsav994 head model -> leadfield matrix is in mat file
def load_fwd_fsav( datafolder:str, head_model_dict:dict, fwd_name:str, scaler_type:str=None ):
    from os.path import expanduser, join
    from mne import convert_forward_solution, read_forward_solution
    # home = expanduser('~')
    model_path = join( 
        datafolder,
        head_model_dict['subject_name'], head_model_dict['orientation'], head_model_dict['electrode_montage'], head_model_dict['source_sampling'], 
        "model" )
    fwd =  read_forward_solution( 
        join( model_path, fwd_name )
    )
    
    leadfield = loadmat( Path( model_path, f"LF_{head_model_dict['source_sampling']}.mat") ) ['G']
    if head_model_dict['orientation'] == 'constrained' : 
        fwd = convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
        )
    fwd['sol']['data'] = leadfield
    if scaler_type == "linear_bis": 
        print("NORMALIZATION LEADFIELD")
        alpha_lf = 10**(find_exp(np.abs(fwd['sol']['data']).max()) + 1)
        fwd['sol']['data'] = fwd['sol']['data'] / alpha_lf
    elif scaler_type == "leadfield": 
        print("NORMALIZATION LEADFIELD")
        fwd['sol']['data'] = fwd['sol']['data'] / (10*np.max(fwd['sol']['data']))
    else : 
        pass
    
    return fwd

def load_mne_info( electrode_montage:str, sampling_freq:int ): 
    # recreate the electrode montage from mne python
    import mne
    if electrode_montage in mne.channels.get_builtin_montages( ): 
        montage = mne.channels.make_standard_montage( electrode_montage )
    elif electrode_montage == "sample":
        montage = make_sample_montage()
    else: 
        sys.exit("Error: unknown electrode montage")

    if electrode_montage == "standard_1020": 
        exclude_mdn = ['T3', 'T4', 'T5', 'T6']
        ids_duplicate = []
        for e in exclude_mdn:
            ids_duplicate.append( np.where( [ch==e for ch in montage.ch_names] )[0][0] )
        ch_names = list( np.delete(montage.ch_names, ids_duplicate) )
            
        info = mne.create_info(
            ch_names, 
            sampling_freq, 
            ch_types='eeg', verbose=False)
    else : 
        info = mne.create_info(
            montage.ch_names, sampling_freq, ch_types='eeg', verbose=None)
        
    info.set_montage(montage)

    return info

## get subset data
def replace_root_fn( simu_path, simu_name, string): 
        """ 
        replace "root" of file name (handle differences in simulations)
        """
        split = string.split(simu_name)
        mod_string = Path( 
            simu_path, simu_name, *split[1].split('/') 
            )
        return mod_string

def get_idx_condition(spos, condi="left-back"):
    # spos = fwd['source_rr']
    px = spos[:,0]
    py = spos[:,1]
    pz = spos[:,2]

    px0 = px.mean() 
    py0= py.mean() 
    pz0 = pz.mean()
    if condi=="left-back":
        x_cond = px < px0 
        y_cond = py < py0 
        all_cond = np.logical_and(x_cond, y_cond)
        selected_pos = spos[all_cond,:]
        index_cond = np.where(all_cond)[0]
    return index_cond

def get_subdataset(condi, datafolder, head_model, dataset_kw, spos, **kwargs): 
    """
    Get index of data which corresponds to a condition condi (condi = seeds verifiying a given condition on there positions)
    """
    # datafolder = cfg.datafolder
    ## build path to simulation ##
    simu_path = Path(  
        datafolder, head_model.subject_name, head_model.orientation, head_model.electrode_montage , head_model.source_sampling, "simu"
            )
    # config_file = Path(
    #     dataset_kw['datafolder'], 
    #     dataset_kw['subject_name'], dataset_kw['orientation'], dataset_kw['electrode_montage'], dataset_kw['source_sampling'], "simu",
    #     dataset_kw['simu_name'], f"{dataset_kw['simu_name']}{dataset_kw['source_sampling']}_config.json"
    #         )
    # with open(config_file) as f : 
    #     config_dict = json.load(f)

    info_file = Path( simu_path, dataset_kw.simu_name ,f"{dataset_kw.simu_name}{head_model.source_sampling}_match_json_file.json" )
    # load info to match the files
    with open(info_file) as f : 
        match_info_dict = json.load(f)
    data_ids = np.array( 
        list( match_info_dict.keys() )
    )
    index_cond = get_idx_condition(condi, spos)

    cond_id = []
    for id in data_ids : 
        md_json_file_name = replace_root_fn( 
            simu_path, 
            dataset_kw.simu_name,
            match_info_dict[id]['md_json_file_name']
            )
        with open( Path(datafolder, md_json_file_name) ) as f : 
            md = json.load(f)


        seed = md['seeds']
        if type(seed) != list: 
            seed = [seed]

        if seed in index_cond : 
            cond_id.append(id)
    
    where_to_save = Path( simu_path, dataset_kw.simu_name, f"{condi}.txt")
    with open(where_to_save, 'w') as f:
        for line in cond_id:
            f.write(f"{line}\n")
    
    return cond_id


import yaml
def plot_source_estimate(src, t_max, fwd, surfer_view="lateral", fs=512, threshold=0.2, hemi="lh", view="lat"):    
    surfer_params_file = "params_surfer.yaml"
    with open(surfer_params_file, "r") as f:
        surfer_kwargs = yaml.safe_load(f)
    surfer_kwargs['hemi'] = hemi
    surfer_kwargs['views'] = view

    j_plot = src[:, t_max]
    if j_plot.max() != 0 : 
        j_plot = j_plot / np.abs(j_plot).max()

    stc_plot = mne.SourceEstimate(
        data=j_plot,
        vertices=[
            fwd["src"][0]["vertno"],
            fwd["src"][1]["vertno"],
        ],
        tmin=0.0,
        tstep=1 / fs,
        subject="fsaverage",
    )
    surfer_kwargs["clim"] = dict(
        kind="value",
        lims=[0, threshold * stc_plot.data.max(), stc_plot.data.max()],
    )
    surfer_kwargs["colorbar"] = False
    # surfer_kwargs["views"] = surfer_view
    brain = stc_plot.plot(**surfer_kwargs)
    img = brain.screenshot()
    brain.close()
    return img

import mne
# give the subject name
def plot_source_estimate_sub(src, t_max, fwd, surfer_view="lateral", fs=512, threshold=0.2, hemi="lh", subject="fsaverage", colorbar=False):    
    surfer_params_file = "params_surfer.yaml"
    with open(surfer_params_file, "r") as f:
        surfer_kwargs = yaml.safe_load(f)
    surfer_kwargs['hemi'] = hemi
    surfer_kwargs['subject'] = subject

    j_plot = src[:, t_max]
    if j_plot.max() != 0 : 
        j_plot = j_plot / np.abs(j_plot).max()

    stc_plot = mne.SourceEstimate(
        data=j_plot,
        vertices=[
            fwd["src"][0]["vertno"],
            fwd["src"][1]["vertno"],
        ],
        tmin=0.0,
        tstep=1 / fs,
        subject=subject,
    )
    surfer_kwargs["clim"] = dict(
        kind="value",
        lims=[0, threshold * stc_plot.data.max(), stc_plot.data.max()],
    )
    surfer_kwargs["colorbar"] = colorbar
    surfer_kwargs["views"] = surfer_view
    if subject=="sample": 
        surfer_kwargs["subjects_dir"] = mne.datasets.sample.data_path() / "subjects"
    brain = stc_plot.plot(**surfer_kwargs)
    img = brain.screenshot()
    brain.close()
    return img


def plot_source_estimate_morph(src, t_max, fwd, fwd_fs, surfer_view="lateral", fs=512, threshold=0.2, hemi="lh", subject="fsaverage"):
    from mne.datasets import sample
    data_path = sample.data_path()
    subjects_dir = Path( data_path, "subjects")
    
    surfer_params_file = "params_surfer.yaml"
    with open(surfer_params_file, "r") as f:
        surfer_kwargs = yaml.safe_load(f)
    surfer_kwargs['hemi'] = hemi
    surfer_kwargs['subject'] = "fsaverage"

    j_plot = src[:, t_max]
    if j_plot.max() != 0 : 
        j_plot = j_plot / np.abs(j_plot).max()

    stc_plot = mne.SourceEstimate(
        data=j_plot,
        vertices=[
            fwd["src"][0]["vertno"],
            fwd["src"][1]["vertno"],
        ],
        tmin=0.0,
        tstep=1 / fs,
        subject=subject,
    )
    ## prepare morphing: 
    src_to = fwd_fs['src']
    morph = mne.compute_source_morph(
        stc_plot,
        subject_from="sample",
        subject_to="fsaverage",
        src_to=src_to,
        subjects_dir=subjects_dir,
        )
    ## apply morphing: 
    stc_fsaverage = morph.apply(stc_plot)

    surfer_kwargs["clim"] = dict(
        kind="value",
        lims=[0, threshold * stc_fsaverage.data.max(), stc_fsaverage.data.max()],
    )
    surfer_kwargs["colorbar"] = False
    surfer_kwargs["views"] = surfer_view
    #if subject=="sample": 
    #    surfer_kwargs["subjects_dir"] = mne.datasets.sample.data_path() / "subjects"
    brain = stc_fsaverage.plot(**surfer_kwargs)
    img = brain.screenshot()
    brain.close()
    return img

def plot_src_from_imgs(imgs,methods, subtitles=None):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    ## plot # todo: enhance the way the colorbar is created to match any threshold value
    fig, axes = plt.subplots(figsize=(16, 5), nrows=1, ncols=len(imgs))
    ii = 0
    cbar_axes = []
    cbar_axes.append(fig.add_axes([0.85, 0.38, 0.01, 0.25]))
    col = plt.cm.Reds(np.linspace(0, 1, 128))
    col_ = [
        (0.0, "#d3d3d3"),
        (0.2, "#fcfafc"),
    ]
    col_ += list(zip(np.linspace(0.2, 1.0, 128), col))
    cmap = LinearSegmentedColormap.from_list("mycmap", col_)

    ii = 0
    for m in methods:
        ax = axes.flat[ii]
        _ = ax.imshow(imgs[m])
        ax.axis("off")
        if subtitles is not None: 
            ax.set_title(m + ":" +subtitles[ii])
        else : 
            ax.set_title(f"{m}")
        ii += 1

    fig.subplots_adjust(right=0.8)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1, clip=False), cmap=cmap),
        cax=cbar_axes[0],
        shrink=0.5,
    )
    plt.show(block=False)

def load_model_from_conf(bsl, baseline_conf):
    """ 
    load one of the baseline model using baseline_conf configuration file
    inputs: 
    - bsl: str: name of the baseline model to load (1dcnn, lstm, deepsif)
    - baseline_conf: omegaconf.dictconfig.DictConfig (or dict): baseline config dict
    output: 
    - net: loaded neural network 
    """
    from contrib.eeg.models_directinv import CNN1Dpl, DeepSIFpl, HeckerLSTMpl
    if bsl.lower() == "1dcnn":
        net = CNN1Dpl(**baseline_conf.get(bsl).model) 
        loaded = torch.load(Path(baseline_conf.get(bsl).path, baseline_conf.get(bsl).name),map_location=torch.device('cpu'))
        # net.model.load_state_dict(loaded)
        if baseline_conf.get(bsl).name[-4:] == "ckpt": 
            net.load_state_dict(loaded['state_dict'])
        else: 
            net.load_state_dict(loaded) 
        net.eval()
    elif bsl.lower() == "lstm": 
        net = HeckerLSTMpl(**baseline_conf.get(bsl).model)
        loaded = torch.load(Path(baseline_conf.get(bsl).path, baseline_conf.get(bsl).name),map_location=torch.device('cpu'))
        # net.model.load_state_dict(loaded)        
        if baseline_conf.get(bsl).name[-4:] == "ckpt": 
            net.load_state_dict(loaded['state_dict'])
        else: 
            net.load_state_dict(loaded)
        net.eval()
    elif bsl.lower() == "deepsif": 
        net = DeepSIFpl(**baseline_conf.get(bsl).model)
        loaded = torch.load(Path(baseline_conf.get(bsl).path, baseline_conf.get(bsl).name),map_location=torch.device('cpu'))#['state_dict']
        if baseline_conf.get(bsl).name[-4:] == "ckpt": 
            net.load_state_dict(loaded['state_dict'])
        else: 
            net.load_state_dict(loaded)
        net.eval()
    else : 
        "unknown baseline"
    return net.eval()

def signal_to_windows(signal, window_length=16, overlap=8, pad=True): 
    """
    Function to convert a signal into an array of time windows of this signal.
    Parameters : 
        - signal : tensor or array of dim [batch_size, channels, time]
        - window_length : size of the window to use
        - overlap : overlap between the time windows
        - pad : wether to pad the signal or not, to get a last time window with zeros completing the time serie points
    Returns : 
        - windows: tensor containing the different time windows, shape [n_time_windows, batch_size, channels, window_length]
    """
    if not torch.is_tensor(signal): 
        signal = torch.from_numpy(signal)
    while len(signal.shape)<3 : 
        signal = signal.unsqueeze(0) #add fake batch and fake channel dimension
    
    n_times = signal.shape[-1]
    step = window_length-overlap
    
    starts = np.arange(0, n_times-window_length, step)

    # for padding :
    if pad : 
        last_win = (n_times-window_length)//step*step
        padding = window_length - (n_times - (last_win + window_length-1)) +1
        padded_signal = torch.zeros((signal.shape[0], signal.shape[1], signal.shape[2]+padding))
        padded_signal[...,:n_times] = signal
        signal = padded_signal
        n_times =padded_signal.shape[-1]
        starts = np.arange(0, n_times-window_length+1, step)
    
    windows = []
    for s in starts: 
        windows.append(signal[...,s:s+window_length].numpy())
    
    return torch.from_numpy( np.stack(windows) )

def windows_to_signal(windows, overlap=8, n_times=None): 
    """
    function to convert an array of windows of dimension [n_windows, batch_size, channels, window_length] into the original time series.
    If overlapping time windows : output the mean of overlapping points
    Parameters : 
        - windows : numpy array of the different time windows
        - overlap : overlap between the time windows
        - n_times : number of time samples to keep in the time series (useful for example if pad was used in the original time series)
    Returns : 
        - signal_recovered : the recovered signal.
    """
    if torch.is_tensor(windows): 
        windows = windows.numpy()
    n_windows = windows.shape[0]
    window_length = windows.shape[-1]
    step = window_length-overlap 
    
    n_times_rec = n_windows*window_length - (n_windows-1)*overlap #- 1 
    signal_recovered = np.zeros( (windows.shape[1], windows.shape[2], n_times_rec) )
    count = np.zeros( (windows.shape[1], windows.shape[2], n_times_rec) )
    starts = np.arange(0, n_times_rec-window_length+1, step)
    for i in range(len(starts)):
        signal_recovered[..., starts[i]:starts[i]+window_length] += windows[i, ...]
        
        count[..., starts[i]:starts[i]+window_length] += 1
    count[count==0] = 1 # avoid division by zero
    signal_recovered/=count
    if n_times : 
        if n_times < n_times_rec:
            return signal_recovered[...,:n_times]
    else : 
        return signal_recovered

def windows_to_signal_center(windows, to_keep=8, overlap=8, n_times=None): 
    """
    function to convert an array of windows of dimension [n_windows, batch_size, channels, window_length] into the original time series.
    If overlapping time windows : output the mean of overlapping points
    Parameters : 
        - windows : numpy array of the different time windows
        - overlap : overlap between the time windows
        - n_times : number of time samples to keep in the time series (useful for example if pad was used in the original time series)
    Returns : 
        - signal_recovered : the recovered signal.
    """
    if torch.is_tensor(windows): 
        windows = windows.numpy()
    n_windows = windows.shape[0]
    window_length = windows.shape[-1]
    step = window_length-overlap 
    
    n_times_rec = n_windows*window_length - (n_windows-1)*overlap #- 1 
    signal_recovered = np.zeros( (windows.shape[1], windows.shape[2], n_times_rec) )
    count = np.zeros( (windows.shape[1], windows.shape[2], n_times_rec) )
    starts = np.arange(0, n_times_rec-window_length+1, step)
    
    for i in range(len(starts)):
        center = starts[i] + window_length // 2
        center_win = window_length // 2

        signal_recovered[..., center-to_keep//2:center+to_keep//2] += windows[i, :,:, center_win-to_keep//2:center_win+to_keep//2]
        # count[..., starts[i]:starts[i]+window_length] += 1

        # count[..., :center-to_keep//2] -= 1
        # count[..., center+to_keep//2:] -= 1
    # count[count==0] = 1 # avoid division by zero
    # signal_recovered/=count
    
    if n_times : 
        if n_times < n_times_rec:
            return signal_recovered[...,:n_times]
    else : 
        return signal_recovered
    
## test focal loss 
def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))

## addition to compute the spatial gradient : 

def compute_adjacency(fwd, model_datafolder=None, source_sampling="fsav_994"): 
    """
    Compute the adjacency matrix of the source space mesh based on the forward object
    Input : 
        fwd : forward object from mne python
    Output : 
        A, the adjacency matrix. Array of shape n_sources x n_sources. A_{i,j} = 1 if source i and j are neighbors in the mesh.
    """
    ## load triangles and vertices then compute neighbors
    

    if source_sampling == "fsav_994": 
        neighbors = np.squeeze( loadmat(f"{model_datafolder}/fs_cortex_20k_region_mapping.mat")['nbs'] )

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
        neighbors = get_neighbors(
            [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
            [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
        )
    
    n_sources = neighbors.shape[0]
    A = np.zeros((n_sources, n_sources)) #adjency matrix
    for i in range(n_sources):
        for n in neighbors[i]: 
            if n >= 0:
                A[i,n] = 1

        A[i,i]=1
    return A

def compute_gradient_matrix(A): 
    """
    Computes the gradient matrix for the adjacency matrix A, based on the edges of the mesh 
    (article de ref : 
    Imaging Brain Source Extent from EEG/MEG by Means of an Iteratively Reweighted Edge Sparsity Minimization (IRES) Strategy - A. Sohrabpour et al)
    """
    ## compute edges : 
    E = []
    n_sources = A.shape[0]
    for i in range(n_sources): 
        for j in range(i+1, n_sources): 
            if A[i,j]==1:
                E.append( (i,j) )
    n_edges = len(E)
    # print(f"number of edges: {len(E)=}")
    ## compute "gradient" matrix : 
    Wg = np.zeros( (n_edges, n_sources) )
    for k, e in enumerate(E): 
        Wg[ k, e[0] ] = 1
        Wg[ k, e[1] ] = -1
    
    return torch.from_numpy(Wg).float()

def compute_spatial_grad( fwd, datafolder=None, source_sampling='fsav_994', gradient_type='laplacian' ): 
    """
    Computes the spatial gradient matrix : edge or laplacian
    """
    adjacency_mat = compute_adjacency(fwd, datafolder, source_sampling)

    if gradient_type.lower()=="edge": 
        gradient_mat = compute_gradient_matrix(adjacency_mat)
    elif gradient_type.lower()=="laplacian": 
        gradient_mat = torch.from_numpy( np.diag( adjacency_mat.sum(0) ) - adjacency_mat ).float()
    else : 
        gradient_mat = None
        sys.exit("Unknown gradient type - available = (edge, laplacians)")
    return gradient_mat

def get_datafolder_path(dataset): 
    return Path(
        dataset['datafolder'], 
        dataset['subject_name'],
        dataset['orientation'],
        dataset['electrode_montage'], 
        dataset["source_sampling"],
        "model"
    )

def reg_to_verts_src(src_reg, n_vertices, region_mapping, fwd_vertices, fs=512): 
    n_regs = np.unique(region_mapping).shape[0]
    src_vertices = np.zeros((n_vertices, src_reg.shape[-1]))
    for r in range(n_regs):
        src_vertices[np.where(region_mapping == r)[0], :] = src_reg[r, :]
    stc = mne.SourceEstimate(
        data=src_vertices,
        vertices=[fwd_vertices["src"][0]["vertno"], fwd_vertices["src"][1]["vertno"]],
        tmin=0.0,
        tstep=1 / fs,
        subject="fsaverage",
    )
    return stc, src_vertices

def get_src_position(src): 
    # retourn used source positions from SourceSpaces object
    spos = np.zeros( (src[0]['nuse']+src[1]['nuse'],3))
    spos[:src[0]["nuse"], :] = src[0][ 'rr' ][src[0]['inuse']==1]
    spos[src[0]["nuse"]:, :] = src[1][ 'rr' ][src[1]['inuse']==1]
    return spos

def get_montage_pos(dig_montage): 
    ch_pos = dig_montage.get_positions()['ch_pos']
    e_pos = np.zeros( (len(ch_pos), 3) )
    for i,(v,p) in enumerate(ch_pos.items()) : 
        e_pos[i,:] = p
    return e_pos


def le_neighb(fwd, neighbors, s, src, src_hat): 
    """
    Localization error in terms of neighbor order
    fwd: fwd object (used to get source positions)
    neighbors: neighbors
    s: ground truth max source 
    src: ground truth source 
    src_hat: estimated source
    """
    # src_hat = torch.from_numpy(src_hat)
    spos = torch.from_numpy(fwd['source_rr'])
    # estimated source = source with max energy: 
    s_hat = torch.argmax(src_hat.sum(1).abs())
    ## is the GT source in the first order neighborhood of the s_hat source?
    le = -1
    o = 0
    while le < 0 and o < 10:
        neighb = get_patch(order=o, idx=s_hat, neighbors=neighbors)
        if s in neighb: 
            le = o
        o += 1
    if le==-1: 
        le = 10
    return le #le in order of neighborhood