# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:01:20 2017

@author: kang927
"""
%load_ext autoreload
%autoreload 2

import sys
import numpy as np
import os
import glob
import re
# for copying data
import shutil

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
mri_img_dn = 'D:/PDFF_roi_training/raw_data/images'
patient_list = listdir_fullpath(mri_img_dn)

Nimg = len(patient_list)
lipoquant_list = []
for ii in range(Nimg):
    # find the subdirectory that matches LQ_6E series
    matched_dn_lst = grab_subdirectory(patient_list[ii], subdirectory_pattern='\\\\LQ_WL')
    if len(matched_dn_lst)!=1:
        print( patient_list[ii] + " has %d lipoquant series"%len(matched_dn_lst) )
    if matched_dn_lst:
        for matched_dn in matched_dn_lst:
            lipoquant_list.append(matched_dn)

print("Found %d image series contain PDFF 6 echoes data"%len(lipoquant_list))


#%%
Nv = len(lipoquant_list)

#%%
dest_dn = 'F:/liverseg_mri6E_training/'
for ii in range(Nv):
    tmplist = glob.glob(os.path.join(lipoquant_list[ii], "*nii"))
    if len(tmplist)==2:
        output_path = dest_dn + 'case_' + str(ii)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for tmpfn in tmplist:
            shutil.copy2(tmpfn,output_path)
            case_fn = output_path + '/'+'case_info.txt'
            with open(case_fn,'w') as f:
                f.write(lipoquant_list[ii])
                f.close()

#%%
