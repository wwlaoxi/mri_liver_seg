# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:05:57 2017

@author: kang927
"""
%load_ext autoreload
%autoreload 2

import sys
import numpy as np
import os
import glob
import re
# for visual displlay of vollume
from matplotlib import pyplot as plt
from IPython import display

from keras.models import load_model 

import nibabel as nib # for reading nifTi data file 
from src.dilatedCNN import dilatedCNN, dice_coef, jacc_dist
# custom modules
from src.utility import listdir_fullpath,grab_subdirectory, resample_imgvol, imshow, write_imgdata_to_nfti
from src.process_dicom import process_dicom, process_dicom_multiEcho 

# the following are for debugging only
from src.process_dicom import read_dicom, sort_dicom_list_multiEchoes

#%%
#To do: make it a python script
#pt_dn = sys.argv[0]
# the directory of the export file


#%% load the CNN model

model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_dcnn_initCT_11082017'
model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_drsn_11162017'
model = load_model(model_fn, custom_objects={'jacc_dist': jacc_dist, 
                                            'dice_coef': dice_coef})

#%%
mri_img_dn = 'D:/liverseg_training/mri_liverseg_LAGB_testcases'
patient_list = listdir_fullpath(mri_img_dn)

Nimg = len(patient_list)
lipoquant_list = []
for ii in range(Nimg):
    # find the subdirectory that matches LQ_6E series
    matched_dn_lst = grab_subdirectory(patient_list[ii], subdirectory_pattern='\\\\LQ')
    if len(matched_dn_lst)!=1:
        print( patient_list[ii] + " has %d lipoquant series"%len(matched_dn_lst) )
    if matched_dn_lst:
        for matched_dn in matched_dn_lst:
            lipoquant_list.append(matched_dn)

print("Found %d image series contain PDFF 6 echoes data"%len(lipoquant_list))

#%%
Nv = len(lipoquant_list)
for ii in range(42,Nv):
    mri_6echo_path=lipoquant_list[ii]
    print("processng image series: "+ str(mri_6echo_path))
    result = process_dicom_multiEcho(mri_6echo_path, target_x_size=224, target_y_size=224,target_z_size=0)
    # image data is Nx-by-Ny-by-Nz-by-Necho data
    # prepare data for segmentation
    echo_num = 0
    # prepare the data for CNN prediction
    img = result['image_data'][:,:,:,echo_num]
    img_data = np.transpose(result['image_data'][:,:,:,echo_num:echo_num+2],[2,0,1,3])

    # get the pdff map
    #pdff_path = 'D:/liverseg_training/mri_training_liverseg/test_cases/Hashem_segmentation/81 ROI ready/Cynch_5413_Hwv_07Jul13/Research_Sirlin_Cynch_Visit_2 - 6427/FFLQ_WL_9016'
    #pdffmap = process_dicom(pdff_path,target_x_size=224, target_y_size=224,target_z_size=0)

    pred = model.predict(img_data,batch_size=8)
    seg_mask = np.transpose(pred[:,:,:,0],[1,2,0])
    img_echo1 = np.transpose(img_data[:,:,:,0],[1,2,0])
    # write out the segmentation as nifti
    fpath = mri_6echo_path
    fname_echo1 = 'img_echo1.nii'
    fname_seg = 'segmentation.nii'
    fname_pdff ='pdffmap.nii'
    write_imgdata_to_nfti(fpath, fname_echo1, img_echo1, result['image_resolution'])
    write_imgdata_to_nfti(fpath, fname_seg, seg_mask, result['image_resolution'])
    #write_imgdata_to_nfti(fpath, fname_pdff, pdffmap, result['image_resolution'])

#%%
pdff_values = pdffmap[seg_mask==1]
plt.hist(pdff_values,bins=70)
pdff_mean = estimate_mean_from_gmm(pdff_values)

#%%
for lipoquant_fn in lipoquant_list:
    print(lipoquant_fn)