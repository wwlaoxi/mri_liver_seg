#!/usr/bin/env python
import numpy as np

# for postprocessing
from scipy.ndimage import label, binary_erosion, binary_dilation 
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
def convert_binary(img_vol):
    binary_vol = np.zeros(img_vol.shape)
    binary_vol[img_vol>=0.5] = 1.0
    binary_vol[img_vol<0.5] = 0.0
    return binary_vol
    
    
#%%
def postprocess_cnn(cnn_seg):
    
    nobj = 3
    struct1 = np.ones((nobj,nobj,nobj))
    pred_postp1 = binary_erosion(cnn_seg,structure=struct1)
    struct2 = np.ones((3,3,3))
    labels, num_features = label(pred_postp1,struct2)
    feature_size = np.zeros((num_features,))   
    for ii in range(num_features):
        feature_size[ii] = np.sum(labels==(ii+1))
    
    label_liver = np.argmax(feature_size)
    pred_postp = np.zeros(cnn_seg.shape)
    pred_postp[labels== (label_liver+1)] = 1
    pred_postp = binary_dilation(pred_postp,struct1)

    return pred_postp