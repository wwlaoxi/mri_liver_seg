# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:44:47 2017

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

#from keras.models import load_model 

import nibabel as nib # for reading nifTi data file 
#from src.dilatedCNN import dilatedCNN, dice_coef, jacc_dist
# custom modules
from src.utility import listdir_fullpath,grab_subdirectory, resample_imgvol, imshow, write_imgdata_to_nfti, copy_image_series
from src.process_dicom import process_dicom, process_dicom_multiEcho 


#%%

# script to extract the lipoquant LAGB data into a new folder in D drive

lagb_dn = 'F:/'
lagb_lst = glob.glob(os.path.join(lagb_dn, "LAGB01*"))

#%%
Nimg = len(lagb_lst)
pdffmap_list = []
for ii in range(Nimg):
    # find the subdirectory that matches LQ_6E series
    matched_dn_lst = grab_subdirectory(lagb_lst[ii], subdirectory_pattern='\\\\FFLQ.*WL')
    if len(matched_dn_lst)!=1:
        print( lagb_lst[ii] + " has %d lipoquant series"%len(matched_dn_lst) )
    if matched_dn_lst:
        for matched_dn in matched_dn_lst:
            pdffmap_list.append(matched_dn)

print("Found %d image series contain PDFF map data"%len(pdffmap_list))


#%%
output_dn='D:/liverseg_training/mri_liverseg_LAGB_testcases/'
num_cases = len(pdffmap_list)
pat_case = re.compile('Lagb[o0]1_[0-9]*_*[A-Za-z]*_[0-9]*')
pat_lq = re.compile('FF') 
for ii in range(num_cases):
    dnlst_to_copy = []
    pdffmap_dn = pdffmap_list[ii]
    # get the case name which is in format Lagb.*
    match_case = pat_case.search(pdffmap_dn)
    case_dn = match_case.group(0)
    print("Create case: " + str( case_dn ))
    # we get the directory where FFmap series 
    src_base_dn = os.path.dirname(pdffmap_dn)
    pdff_dn = os.path.basename(pdffmap_dn)

    # the second part to convert to interger first because series number is 9 if ffmap series is 9009
    lq_dn = pdff_dn[2:-4] + str( np.int(pdff_dn[-2:]))
    # check whether lipoquant directory exit:
    tmp = src_base_dn+'/'+lq_dn
    if os.path.isdir(tmp):
        print( "lipoquant image series exist: " + str(tmp) )
        dnlst_to_copy.append(pdff_dn)
        dnlst_to_copy.append(lq_dn)
        # now we can create the desire directories
        for dn_to_copy in dnlst_to_copy:
            src_fullpath_dn = src_base_dn +'/' +dn_to_copy
            dest_fullpath_dn = output_dn + '/' + case_dn + '/' + dn_to_copy
            #print("copy %s to %s"%(src_fullpath_dn,dest_fullpath_dn))
            copy_image_series(src_fullpath_dn, dest_fullpath_dn)
        
    else:
        print("lipoquant image series NOT EXIST!")
    
    #output_path = output_dn +  case_dn +'/'+os.path.basename(lipoquant_dn)
    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)
    
    #copy_image_series(lipoquant_dn, output_path)
    
    
    
#%%
