# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 07:29:58 2017

@author: kang927
"""

%load_ext autoreload
%autoreload 2

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import re
from xlrd import open_workbook
from scipy.ndimage.interpolation import zoom
from  natsort import natsorted
import dicom


from src.process_dicom import process_dicom, process_dicom_multiEcho
from src.utility import listdir_fullpath,grab_subdirectory, imshow


#%%
def save_img_slice(datavol,file_prefix):
    nslice = datavol.shape[2]
    for s in range(nslice):
        fn = file_prefix + '_slice_' + str(s) + '.npy'
        np.save( fn,datavol[:,:,s])
        
#%%
# extract all the lipoquant image series
mri_img_dn = 'D:/liverseg_training/mri_training_liverseg/EZE_liver_volume_segmentations'
patient_list = listdir_fullpath(mri_img_dn)

Nimg = len(patient_list)
lipoquant_img_list = []
lipoquant_seg_list =[]

for ii in range(Nimg):
    # find the subdirectory that matches LQ_6E series
    matched_dn = grab_subdirectory(patient_list[ii], subdirectory_pattern='LQ')
    matched_dn2 = grab_subdirectory(patient_list[ii], subdirectory_pattern='segmentation_')
    if len(matched_dn)!=1:
        print( patient_list[ii] + " has %d lipoquant series"%len(matched_dn) )
    if matched_dn:
        lipoquant_img_list.append(matched_dn)
    if matched_dn2:
        lipoquant_seg_list.append(matched_dn2)
        
        
        
print("Found %d image series contain PDFF 6 echoes data"%len(lipoquant_img_list))
print("Found %d image series contain PDFF seg data"%len(lipoquant_seg_list))
#for fn  in lipoquant_list:
#    print(fn)

#%%
Nvol = len(lipoquant_img_list)

img_list =[]
mask_list =[]

for ii in range(Nvol):
    #ii = ind_goodcase[kk]
    path  = lipoquant_img_list[ii][0]
    result = process_dicom_multiEcho(path, target_x_size=224, target_y_size=224,target_z_size=0)    
    img_data = result['image_data']
    # now need to load the segmentation mask
    path_seg = lipoquant_seg_list[ii][0]
    seg_tmp = process_dicom(path_seg, target_x_size=224, target_y_size=224, target_z_size =0)
    mask = np.zeros(seg_tmp.shape)
    mask[seg_tmp>=2000] = 1                
    s=10
    print('case '+ str(ii))
    imshow(img_data[:,:,s,0],mask[:,:,s])
    img_list.append(img_data)
    mask_list.append(mask)




#%%
train_img_dir ='D:/liverseg_training/mri_training_liverseg/training/images/';
train_seg_dir ='D:/liverseg_training/mri_training_liverseg/training/segmentations/';

jj=84
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
           
    
    
    