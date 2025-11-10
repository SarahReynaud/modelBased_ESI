"""   
functions for inverse problem resolution
"""

import mne
import numpy as np

#########" compute covariance

def mne_compute_covs( M, mne_info, activity_thresh=0.1 ): 
    """  
    Compute noise and data covariance of EEG signal M. 
    Active signal (for data covariance computation) are the points of the signals for which the activity is superior 
    to activity_thresh * amplitude max

    Returns noise_cov and data_cov, covariance object from mne-python
    """
    fs = mne_info["sfreq"]
    evoked_eeg = mne.EvokedArray( M , mne_info, verbose=None )
    e_max, t_max = evoked_eeg.get_peak(time_as_index=True)
    e_max = int( np.squeeze( np.where( [ chn==e_max for chn in mne_info['ch_names'] ] ) ) ) -1
    #e_max = int(e_max) - 1
    amp_max = evoked_eeg.get_data()[e_max, t_max] 

    all_points = np.arange( 0, M.shape[1], 1 )
    active_points = np.where( np.abs( evoked_eeg.get_data()[e_max,:] )> activity_thresh*np.abs( amp_max ) )[0].squeeze()
    noise_points = np.delete( all_points, active_points) 
    
    n_active_points =  len( active_points ) 

    if n_active_points > 16 and (n_active_points < M.shape[1]-100) :
        data_signal     = mne.io.RawArray( M[:,active_points], mne_info, verbose=False)
        data_signal.set_eeg_reference(verbose=False)
        data_cov = mne.compute_raw_covariance( 
            data_signal, tmin=0., tmax=None, tstep=1/fs, method="auto", rank=None, verbose=False
        )
    elif (n_active_points > M.shape[1]-100): 
        data_cov = None 
        noise_points = all_points 
    else: 
        data_cov = None
    
    noise_signal    = mne.io.RawArray( M[:,noise_points], mne_info, verbose=False)
    noise_signal.set_eeg_reference(verbose=False)
    noise_cov = mne.compute_raw_covariance(  
        noise_signal, tmin=0., tmax=None, tstep = 1/fs, method="auto", rank=None, verbose=False
    )
    
    return noise_cov, data_cov, n_active_points




# Minimum norm #
# raw data
def gen_mn_raw( raw, noise_cov,fwd, lambda2 , method='sLORETA'):
    # gen_mn_raw _______________________
    # Function to compute the inverse solution using a minimum norm method, on
    # data of type "Raw"
    # 
    # Inputs : 
    # - raw : Raw object containing the eeg data of interest
    # - noise_co : Covariance object of the noise covariance of the signal
    # - fwd : Forward object of the forward model corresponding to the data
    # - lambda2 : regularisation parameter
    # - method : method to use : can be sLORETA, dSPM, mne, eLORETA
    # Output : 
    # -mn_stc : SourceEstimate object, result of the inversion
    #____________________________________
    #if not volume : 
    # Create inverse operator 
    raw.set_eeg_reference(projection=True, verbose=False)
    inv_operator = mne.minimum_norm.make_inverse_operator( raw.info,\
        fwd, noise_cov, loose=0, depth=0, fixed=True, verbose = False )
    #apply inverse operator on data
    mn_stc = mne.minimum_norm.apply_inverse_raw( raw, inv_operator, lambda2, \
        method=method, pick_ori=None, verbose=False)
    '''
    else : 
        # Create inverse operator
        inv_operator = mne.minimum_norm.make_inverse_operator( raw.info,\
            fwd, noise_cov, loose=1, depth=0, fixed='auto', verbose = False )
        # Apply inverse operator on data
        mn_stc = mne.minimum_norm.apply_inverse_raw( raw, inv_operator, lambda2, \
            method=method, pick_ori='vector', verbose=False)
    '''
    return mn_stc 

def gen_mn_evoked( evoked, noise_cov, fwd, lambda2 , method='sLORETA'):
    # gen_mn_raw _______________________
    # Function to compute the inverse solution using a minimum norm method, on
    # data of type "Raw"
    # 
    # Inputs : 
    # - raw : Raw object containing the eeg data of interest
    # - noise_co : Covariance object of the noise covariance of the signal
    # - fwd : Forward object of the forward model corresponding to the data
    # - lambda2 : regularisation parameter
    # - method : method to use : can be sLORETA, dSPM, mne, eLORETA
    # Output : 
    # -mn_stc : SourceEstimate object, result of the inversion
    #____________________________________
    #if not volume : 
    # Create inverse operator 
    evoked.set_eeg_reference(projection=True, verbose=False)
    inv_operator = mne.minimum_norm.make_inverse_operator( evoked.info,\
        fwd, noise_cov, loose=0, depth=0, fixed=True, verbose = False )
    #apply inverse operator on data
    mn_stc = mne.minimum_norm.apply_inverse( evoked, inv_operator, lambda2, \
        method=method, pick_ori=None, verbose=False)
    '''
    else : 
        # Create inverse operator
        inv_operator = mne.minimum_norm.make_inverse_operator( raw.info,\
            fwd, noise_cov, loose=1, depth=0, fixed='auto', verbose = False )
        # Apply inverse operator on data
        mn_stc = mne.minimum_norm.apply_inverse_raw( raw, inv_operator, lambda2, \
            method=method, pick_ori='vector', verbose=False)
    '''
    return mn_stc 

# Beamformer
def gen_beamf_raw( raw_EEG, data_cov, fwd, lambda2, noise_cov=None, constrained=True, volume=False):
    raw_EEG.set_eeg_reference(projection=True, verbose=False)
    if constrained: 
        pick_ori = None
    else:
        if volume : 
            pick_ori = "vector" 
        else : 
            pick_ori = "max-power"

    lcmv_filters = mne.beamformer.make_lcmv(raw_EEG.info, fwd, data_cov, reg=lambda2, pick_ori=pick_ori,
        reduce_rank=False, verbose = False) # avec rank=None résulat faut ?? pourtant c'est la valeur par défaut... LOUCHE; weight_norm='unit-noise-gain'
    lcmv_stc     = mne.beamformer.apply_lcmv_raw(raw_EEG, lcmv_filters, verbose = False)
    return lcmv_stc



