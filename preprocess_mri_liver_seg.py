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
from xlrd import open_workbook
from scipy.ndimage.interpolation import zoom
from  natsort import natsorted
import dicom

# custom modules
from src.utility import listdir_fullpath,grab_subdirectory, resample_imgvol, imshow
from src.process_dicom import process_dicom, process_dicom_multiEcho

#%%
def save_img_slice(datavol,file_prefix):
    nslice = datavol.shape[2]
    for s in range(nslice):
        fn = file_prefix + '_slice_' + str(s) + '.npy'
        np.save( fn,datavol[:,:,s])
        

#%%
# extract all the lipoquant image series
mri_img_dn = 'D:/liverseg_training/mri_training_liverseg/images/'
patient_list = listdir_fullpath(mri_img_dn)

Nimg = len(patient_list)
lipoquant_list = []
for ii in range(Nimg):
    # find the subdirectory that matches LQ_6E series
    matched_dn = grab_subdirectory(patient_list[ii], subdirectory_pattern='LQ_6E')
    if len(matched_dn)!=1:
        print( patient_list[ii] + " has %d lipoquant series"%len(matched_dn) )
    if matched_dn:
        lipoquant_list.append(matched_dn)

print("Found %d image series contain PDFF 6 echoes data"%len(lipoquant_list))
#for fn  in lipoquant_list:
#    print(fn)


#%%
Nvol = len(lipoquant_list)
mri_seg_dn = 'D:/liverseg_training/mri_training_liverseg/segmentations/'
seg_list = listdir_fullpath(mri_seg_dn)

#%%
img_list =[]
mask_list =[]
for ii in range(Nvol):
    path  = lipoquant_list[ii][0]
    result = process_dicom_multiEcho(path, target_x_size=224, target_y_size=224,target_z_size=0)    
    img_data = result['image_data']
    # the segmentation mask is identify by patient ID
    # search for the segmentation mask directroy by patient ID
    pat = re.compile(result['patientID']+'_'+result['AcquisitionDate'])
    for segfile in seg_list:
        if re.search(pat,segfile):
            result2=process_dicom_multiEcho(segfile, target_x_size=224, target_y_size=224,target_z_size=0)
            # unclear why, but for some series, the segmentation files have more axial slices than the images
            if result2['last_instance_number'] <= result['last_instance_number']:
                print("found segmentation file matching patient ID: "+result['patientID'])
                img_data2=result2['image_data']
                mask = np.zeros(img_data.shape)
                # instance number start at 1 but python array start at 0
                start_index =np.int( np.ceil(result2['first_instance_number']/result2['num_echoes']) ) - 1
                end_index = np.int( start_index + img_data2.shape[2])
                tmp = img_data2
                # convert the image to binary data 
                tmp[tmp!=0] = 1
                mask[:,:,start_index:end_index] = tmp
                # add both images and segmentation mask into the list
                img_list.append(img_data)
                mask_list.append(mask)


#%% visualize the data
Nz = img_data.shape[2]
for img, mask in zip(img_list, mask_list):
    for ii in range(0,img.shape[2],10):
        echo = 3
        imshow(img[:,:,ii,echo-1],mask[:,:,ii,echo-1])
   
#%%
train_img_dir ='D:/liverseg_training/mri_training_liverseg/training/images/';
train_seg_dir ='D:/liverseg_training/mri_training_liverseg/training/segmentations/';

jj=0
for img, mask in zip(img_list,mask_list):
    # save data per slice
    for ii in range(img.shape[3]):
        echo_num = ii+1
        img_data = img[:,:,:,ii]
        seg_data = mask[:,:,:,ii]
        img_prefix=train_img_dir + 'image' + str(jj) + '_echo' + str(ii) 
        seg_prefix=train_seg_dir + 'seg' + str(jj) + '_echo' + str(ii) 
        save_img_slice(img_data,img_prefix)
        save_img_slice(seg_data,seg_prefix)
    jj = jj+1
        
        
        