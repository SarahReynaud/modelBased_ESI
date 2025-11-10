""" 
jan 23.
functions linked with results evaluation """
# See nb_eval_metrics for other metrics. for now not really used.

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

######################################################################
######################## MSE, nMSE ##################################
#####################################################################

def mse_fn( x, x_hat ): 
    """  
    Mean Squared Error (MSE) between x and x_hat
    """
    assert x.shape==x_hat.shape 
    mse_val = ( ( x - x_hat )**2 ).mean()
    return mse_val

def batch_mse_fn( x, x_hat ): 
    """ 
    Mean Squared Error on batch x and x_hat (same as MSE on 2D or 1D data)
    """
    return mse_fn(x, x_hat)


def nmse_t_fn( x, x_hat, t):
    """  
    normalised mean squared error (nMSE) between x and x_hat, at time t.
    x and x_hat are excpected to be 2 dimensional vectors

    return the nMSE value.
    """
    assert  x.shape == x_hat.shape
    assert len(x.shape)==2

    x       = x[:,t].squeeze()
    x_hat   = x_hat[:,t].squeeze()

    if x.max() == 0. or x_hat.max()==0. : 
        x_n = x
        x_hat_n = x_hat
    else:
        x_n = x / x.abs().max()
        x_hat_n = x_hat / x_hat.abs().max()

    #nmse_t_val = (1/x_n.shape[0]) *( ( x_n - x_hat_n )**2 ).sum()
    nmse_t_val =  ( ( x_n - x_hat_n )**2 ).mean()

    return nmse_t_val

def nmse_fn(x, x_hat): 
    """  
    normalized Mean Squared Errro between x and x_hat (on both dimensions)
    x and x_hat are excpected to be 2 dimensional vectors
    """
    assert x.shape==x_hat.shape 
    assert len(x.shape)==2

    if x.max()==0. : 
        x_n = x_n
    else:
        x_n         = x / x.max()

    if x_hat.max()==0.: 
        x_hat_n = x_hat
    else:    
        x_hat_n     = x_hat / x.max() 

    return mse_fn(x_n, x_hat_n)

def batch_nmse_fn(x, x_hat): 
    """  
    normalized Mean Squared Errro for a batch x and a batch x_hat.
    x and x_hat are excpected to be 3 dimensional vectors where the first dimension
    is the batch dimension B. 
    Returns the mean over the batches.
    """
    assert x.shape==x_hat.shape 
    assert len(x.shape)==3

    max_scaler  = x.view(x.shape[0], -1).max(dim =  1)[0]
    nul_id = torch.argwhere(max_scaler==0.).squeeze()
    max_scaler[nul_id] = 1
    x_n         = x / max_scaler.view(x.shape[0], 1, 1)

    max_scaler  = x_hat.view(x_hat.shape[0], -1).max(dim =  1)[0]
    nul_id = torch.argwhere(max_scaler==0.).squeeze()
    max_scaler[nul_id] = 1   
    x_hat_n     = x_hat / max_scaler.view(x.shape[0], 1, 1)

    del max_scaler 

    return mse_fn(x_n, x_hat_n)




############################################################################
########################### AUC ############################################
#############################################################################
def auc_t( x_gt, x_hat, t, thresh=False, act_thresh=0.1, act_src=None):
    """    
    - x_gt: ground truth source distribution
    - x_hat: estimated source data
    - time instant to study
    - thresh: whether the thresholding of active sources is based on a threshold (thresh=True) or on a list of active source indices (thresh=False)
    - act_thresh: threshold to determine active source VS inactive sources (purcentage of the maximum amplitude), if thresh=True
    - act_src: list of active source indices, if thresh=False
    """ 
    x_gt    = x_gt.squeeze() 
    x_hat   = x_hat.squeeze()
    assert len(x_gt.shape) == 2 
    assert len(x_gt.shape) == len(x_hat.shape)
    
    x_gt    = x_gt[:,t] 
    x_hat   = x_hat[:,t]
    # get index of active sources: 
    if thresh: 
        Sa = torch.argwhere( x_gt.abs() > act_thresh * x_gt.abs().max()  )
    else: 
        Sa = act_src
    # binarize the ground truth data: active source = 1, other sources = 0
    bin_gt       = torch.zeros_like(x_gt, dtype=int)
    bin_gt[Sa] = 1

    # scale estimated data
    if x_hat.abs().max()==0.: 
        x_hat_unit = x_hat.abs()
    else: 
        x_hat_unit = x_hat.abs()/x_hat.abs().max()
    
    fpr, tpr, _ = roc_curve(bin_gt, x_hat_unit)
    auc_value   = auc(fpr, tpr)

    return auc_value

#######################################################################
################ LOC ERROR ############################################
#######################################################################