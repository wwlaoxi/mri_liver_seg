# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:08:06 2017

@author: kang927
"""

%load_ext autoreload
%autoreload 2

import numpy as np
import os
import glob
# for visual displlay of vollume
from matplotlib import pyplot as plt
from IPython import display
# use for natural order sorting
from natsort import natsorted
import nibabel as nib # for reading nifTi data file
import re
from keras.models import load_model
# custom function 
from src.dilatedCNN import dilatedCNN, dice_coef_loss, dice_coef, jacc_dist
from src.dilatedresidualCNN import ResnetBuilder
from src.metrics import dice_coef_py, vol_overlap_err, rel_vol_diff, convert_binary, postprocess_cnn, estimate_mean_from_gmm
from src.utility import listdir_fullpath,grab_subdirectory, imshow, write_imgdata_to_nfti, preprocess_segvol, get_FFmap_fn
from src.process_dicom import process_dicom, process_dicom_multiEcho, normalize_image 
from src.get_Couinaud_segment import get_centroid_coordinate, get_bisecting_plane_nl_vec, voxelCoord_to_mask, generate_couinaud_segments
#%% load the model based on all echoes data 
model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_drsn_augment_histeq_individualEcho_12227017'
model = load_model(model_fn, custom_objects={'jacc_dist': jacc_dist, 
                                            'dice_coef': dice_coef})


#%%
Nvol=5
segmentalPDFF_mean = np.zeros((Nvol,9))
segmentalPDFF_std = np.zeros((Nvol,9))
seg_labels = np.array([1,2,3,4,9,5,6,7,8])
#%%
data_dn = 'D:/liverseg_training/couinaud_segments/cases'
case_list = listdir_fullpath(data_dn)
Nv = len(case_list)
for ii in range(0,5):
    case_dn = case_list[ii]
    lq_dn = grab_subdirectory(case_dn,'\\\\LQ')[0]
    ffmap_dn = grab_subdirectory(case_dn,'\\\\FFLQ')[0]
    result = process_dicom(lq_dn, target_x_size=224, target_y_size=224,target_z_size=0)
    img_data = np.transpose( np.array([ result['image_data'] ]), [3,1,2,0])
    # normalize the data from 0 to 1
    img_data = normalize_image( img_data )    
    #img_data1 = img_data[:,:,:,2:3]
    pred = model.predict(img_data,batch_size=8)
    tmp = np.transpose(pred[:,:,:,0],[1,2,0])
    # convert to binary and remove other spurious elements in segmentation
    tmp2 = postprocess_cnn( convert_binary(tmp) )
    liver_seg = tmp2.astype(np.float)
    # loading in the landmark data
    landmark_fn = case_dn + '/2point_seg.nii.gz'
    landmarkObj = nib.load(landmark_fn)
    landmark_tmp = landmarkObj.get_data()
    landmark = np.transpose( landmarkObj.get_data(),[1,0,2])
    # process it to match the image data view
    landmark = np.flip(landmark,axis=2)
    couinaud_seg = generate_couinaud_segments(landmark, liver_seg)
    fname_seg = 'couinaud_seg.nii'
    tmp = np.flip(np.flip(couinaud_seg,axis=0),axis=1)
    write_imgdata_to_nfti(case_dn, fname_seg, tmp, (1.0,1.0,1.0))
    # get the segmental PDFF values from PDFF map
    result2 = process_dicom(ffmap_dn,target_x_size=224, target_y_size=224,target_z_size=0)
    ffmap = result2['image_data']
    # go through each segment
    for jj in range(0,9):
        ff_segments = ffmap[ couinaud_seg== seg_labels[jj] ]
        segmentalPDFF_mean[ii,jj] = np.mean(ff_segments)
        segmentalPDFF_std[ii,jj] = np.std(ff_segments)
#%%
for jj in range(0,9):
    print(segmentalPDFF_mean[3,jj])