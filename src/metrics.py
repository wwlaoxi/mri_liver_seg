#!/usr/bin/env python
from numpy import np


#%%  metrics to evaluate the segmentation volumes, need to also implement surface metrics 
def dice_coef_py(y_true, y_pred):

    smooth = 1.

    y_true_f = y_true.flatten()

    y_pred_f = y_pred.flatten()

    intersection = np.sum( y_true_f * y_pred_f )
    return (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

#%%
def vol_overlap_err(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum( y_true_f * y_pred_f ).astype(np.float64)
    union = np.sum( (y_true_f + y_pred_f)!=0.0 ).astype(np.float64)
    voe = (1 - intersection/union)*100
    return voe


#%% 
def rel_vol_diff(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    vol1 = np.sum(y_true_f)
    vol2 = np.sum(y_pred_f)
    rvd = np.abs( vol1 - vol2)/vol1*100
    return rvd     


#%%