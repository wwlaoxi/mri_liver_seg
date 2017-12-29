#!/usr/bin/env python
import numpy as np

# for postprocessing
from scipy.ndimage import label, binary_erosion, binary_dilation 
from sklearn import mixture

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
    
#%% estimate the whole-liver mean PDFF values assuming there are separate populations of PDFF values: 1) liver parenchyma, 2) vessels 
# can extend to other classes if needed   
def estimate_mean_from_gmm(pdff_values):
    """
    we are assuming the possibility of two group of values, 1 represent PDFF and another represent the blood vessels
    
    """
    pdff_values = pdff_values.reshape(-1,1)
    # use BIC criteria to choose 1 vs. 2 component GMM
    bic = []
    lowest_bic = np.infty
    n_components_range = range(1,3)
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(pdff_values)
        bic.append(gmm.bic(pdff_values))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    labels = best_gmm.predict(pdff_values)
    group_size = np.zeros((2,))
    group_size[0] = np.sum(labels==0)
    group_size[1] = np.sum(labels==1)
    #group_size[2] = np.sum(labels==2)
    return gmm.means_[np.argmax(group_size)]                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               