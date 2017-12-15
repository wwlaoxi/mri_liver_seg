# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:36:44 2017

@author: kang927
"""
%load_ext autoreload
%autoreload 2

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import re
from  natsort import natsorted
import nibabel as nib # for reading nifTi data file
# custom modules
from src.utility import listdir_fullpath,grab_subdirectory, resample_imgvol, imshow, transform_ITKSNAP_to_training
from src.process_dicom import process_dicom, process_dicom_multiEcho

#%%
def save_img_slice(datavol,file_prefix):
    nslice = datavol.shape[2]
    for s in range(nslice):
        fn = file_prefix + '_slice_' + str(s) + '.npy'
        np.save( fn,datavol[:,:,s])
        

#%%
seg_mask_dn = 'D:/liverseg_training/liverseg_mri6E_training'
case_list = listdir_fullpath(seg_mask_dn)

#%%
Nvol = len(patient_list)

#%%
img_list =[]
mask_list =[]
for ii in range(0,Nvol):
    try:
        seg_mask_fn = case_list[ii] +'/manual_segmentation.nii.gz'
        case_info_fn = case_list[ii] +'/case_info.txt'
        fnobj = open(case_info_fn,'r')
        img_path  = fnobj.read()
        fnobj.close()
        result = process_dicom_multiEcho(img_path, target_x_size=224, target_y_size=224,target_z_size=0)    
        img_data = result['image_data']
        # add both images and segmentation mask into the list
        fnobj = nib.load(seg_mask_fn)
        tmp = fnobj.get_data()
        seg_mask = transform_ITKSNAP_to_training(tmp)
        img_list.append(img_data)
        mask_list.append(seg_mask)
    except:
        pass 

#%% visualize the data
for img, mask in zip(img_list, mask_list):
    for ii in range(8,img.shape[2],8):
        echo = 3
        imshow(img[:,:,ii,0],mask[:,:,ii])
   
#%%
train_img_dir ='D:/liverseg_training/mri_training_liverseg/training/images/';
train_seg_dir ='D:/liverseg_training/mri_training_liverseg/training/segmentations/';

jj=174
for img, mask in zip(img_list,mask_list):
    # save data per slice
    seg_data = mask
    for ii in range(img.shape[3]):
        img_data = img[:,:,:,ii]
        img_prefix=train_img_dir + 'image' + str(jj) + '_echo' + str(ii) 
        seg_prefix=train_seg_dir + 'seg' + str(jj) + '_echo' + str(ii) 
        save_img_slice(img_data,img_prefix)
        save_img_slice(seg_data,seg_prefix)
    jj = jj+1
           