# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:23:07 2017

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
from src.metrics import dice_coef_py, vol_overlap_err, rel_vol_diff, convert_binary, postprocess_cnn
from src.utility import listdir_fullpath,grab_subdirectory, resample_imgvol, imshow, write_imgdata_to_nfti, preprocess_segvol
from src.process_dicom import process_dicom, process_dicom_multiEcho 
# for postprocessing
#from scipy.ndimage import label, binary_erosion, binary_dilation 

#%% load the model based on all echoes data 
model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_drsn_augment_histeq_11212017'
model = load_model(model_fn, custom_objects={'jacc_dist': jacc_dist, 
                                            'dice_coef': dice_coef})
#%%

segmentations_dn = 'D:/liverseg_training/mri_liverseg_LAGB_testcases/segmentations/'
img_lst = 'D:/liverseg_training/mri_liverseg_LAGB_testcases/data/'
data_list = listdir_fullpath(segmentations_dn)

Nvol = len(data_list)
# metric use to evaluate segmentations 
diceScore = np.zeros((Nvol,))
voe = np.zeros((Nvol,))
rvd = np.zeros((Nvol,))
liverVol_seg = np.zeros((Nvol,))
liverVol_pred = np.zeros((Nvol,))
#%%
for ii in range(0,Nvol):
    manual_seg_fn = glob.glob(os.path.join(data_list[ii], "*nii.gz"))[0]
    info_fn = data_list[ii] + '/case_info.txt'
    fn_obj = open(info_fn,'r')
    tmp = fn_obj.read()
    fn_obj.close()
    pat = re.compile('Lagb.*')
    matchObj = pat.search(tmp)
    data_dn = matchObj.group(0)
    mri_6echo_path = img_lst + data_dn
    result = process_dicom_multiEcho(mri_6echo_path, target_x_size=224, target_y_size=224,target_z_size=0)
    img_data = np.transpose(result['image_data'][:,:,:,:],[2,0,1,3])
    pred = model.predict(img_data,batch_size=8)
    tmp = np.transpose(pred[:,:,:,0],[2,1,0])
    seg_mask = np.flip( np.flip( np.flip(tmp,axis=2), axis=1), axis=0)
    seg_mask = convert_binary(seg_mask)
    seg_mask = postprocess_cnn(seg_mask)
    fnobj = nib.load(manual_seg_fn)
    manual_seg_mask = fnobj.get_data()
    diceScore[ii] = dice_coef_py(manual_seg_mask,seg_mask)
    voe[ii]= vol_overlap_err(manual_seg_mask,seg_mask)
    rvd[ii] = rel_vol_diff(manual_seg_mask,seg_mask) 
    liverVol_seg[ii] = np.sum(manual_seg_mask.flatten())
    liverVol_pred[ii] = np.sum(seg_mask.flatten())
    # save the segmentation 
    fname_seg = 'segmentation.nii'
    write_imgdata_to_nfti(data_list[ii], fname_seg, seg_mask.astype(np.float), (1.0,1.0,1.0))
    
#%% print summary
print(" mean voe = " + str(np.mean(voe)))
print(" mean dice score = " + str(np.mean(diceScore)))
print(" mean rvd = " + str(np.mean(rvd)))
#%%
plt.figure()
plt.boxplot(diceScore[diceScore>0.3], 0, 'rs', 0, 0.75)
plt.figure()
plt.hist(diceScore[diceScore>0.3], bins=20)

#%%
plt.figure()
plt.boxplot(voe[diceScore>0.5], 0, 'rs', 0, 0.75)
plt.figure()
plt.hist(voe[diceScore>0.5], bins=20)
#%%
plt.figure()
plt.boxplot(rvd[diceScore>0.5], 0, 'rs', 0, 0.75)
plt.figure()
plt.hist(rvd[diceScore>0.5], bins=20)
#%%

# if exclude outliers 
ind = np.where(diceScore>=0.1)

# scatter plot and correlation coefficient between volume measurements 
x = liverVol_seg[ind]
y = liverVol_pred[ind]
pearR = np.corrcoef(x,y)[1,0]
print(pearR.shape)
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y, color='red')
plt.plot(x,x*m+c,color='blue',label='r=%3.3f'%(pearR))
plt.legend(loc=2) # legend at upper left
plt.xlabel('Automatic liver volumes (cm^3)')
plt.ylabel('Manual liver volumes (cm^3)')

#%%
x1 = liverVol_seg
x2 = liverVol_pred

# Bland-altman fashion plot
mean_vol = (x1 + x2)/2
diff_vol = (x1 - x2)/x1*100
md        = np.mean(diff_vol)       # Mean of the difference
sd        = np.std(diff_vol, axis=0)    # Standard deviation of the difference
plt.scatter(mean_vol, diff_vol)
plt.axhline(md,           color='red', linestyle='--')
plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
plt.xlabel('mean Liver Volume (cm^3)')
plt.ylabel('Percent Difference in liver volume (%) ')
plt.title('Bland-Altman Plot of Liver Volume Measurements')
