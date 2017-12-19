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
from src.metrics import dice_coef_py, vol_overlap_err, rel_vol_diff, convert_binary, postprocess_cnn, estimate_mean_from_gmm
from src.utility import listdir_fullpath,grab_subdirectory, imshow, write_imgdata_to_nfti, preprocess_segvol, get_FFmap_fn
from src.process_dicom import process_dicom, process_dicom_multiEcho 
# for postprocessing
#from scipy.ndimage import label, binary_erosion, binary_dilation 

#%% load the model based on all echoes data 
model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_drsn_augment_histeq_11212017'
model = load_model(model_fn, custom_objects={'jacc_dist': jacc_dist, 
                                            'dice_coef': dice_coef})


#%%

    
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
meanPDFF_pred = np.zeros((Nvol,))

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
    # postprocessing 
    # swap the z-axis to the last dimension
    tmp = np.transpose(pred[:,:,:,0],[1,2,0])
    # convert to binary and remove other spurious elements in segmentation
    tmp2 = postprocess_cnn( convert_binary(tmp) )
    tmp2 = tmp2.astype(np.float)
    
    # these processing below is to match the processing done in the manual segmentation mask to calculate accurate volumes
    tmp3 = np.transpose(tmp2,[1,0,2])
    seg_mask = np.flip( np.flip( np.flip(tmp3,axis=2), axis=1), axis=0)
    fnobj = nib.load(manual_seg_fn)
    manual_seg_mask = fnobj.get_data()
    diceScore[ii] = dice_coef_py(manual_seg_mask,seg_mask)
    voe[ii]= vol_overlap_err(manual_seg_mask,seg_mask)
    rvd[ii] = rel_vol_diff(manual_seg_mask,seg_mask) 
    liverVol_seg[ii] = np.sum(manual_seg_mask.flatten())
    liverVol_pred[ii] = np.sum(seg_mask.flatten())
    # save the segmentation 
    #fname_seg = 'segmentation.nii'
    #write_imgdata_to_nfti(data_list[ii], fname_seg, tmp2, (1.0,1.0,1.0))

    # calculation of ffmap
    # get the ffmap file
    ffmap_fn = get_FFmap_fn(data_dn)
    ffmap_path = img_lst + ffmap_fn
    ffmap = process_dicom(ffmap_path,target_x_size=224, target_y_size=224,target_z_size=0)
    ff_liver = ffmap[tmp2==1]
    meanPDFF_pred[ii] = estimate_mean_from_gmm(ff_liver)
    
#%% print summary
print(" mean voe = " + str(np.mean(voe)))
print(" mean dice score = " + str(np.mean(diceScore)))
print(" mean rvd = " + str(np.mean(rvd)))
#%%
#plt.figure()
#plt.boxplot(diceScore[diceScore>0.1], 0, 'rs', 0, 0.75)
plt.figure()
fig = plt.figure()
ax = fig.add_subplot(1,1,1,)
ax.hist(diceScore[diceScore>0.1], bins=20)
plt.xlabel('DICE  score')
plt.ylabel('Numbers of image sets')

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
x = liverVol_seg[ind]/33.3
y = liverVol_pred[ind]/33.3
pearR = np.corrcoef(x,y)[1,0]
print(pearR.shape)

#%%
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y, color='red')
plt.plot(x,x*m+c,color='blue',label='r=%3.3f'%(pearR))
plt.legend(loc=2) # legend at upper left
plt.xlabel('Automatic liver volumes (cm^3)')
plt.ylabel('Manual liver volumes (cm^3)')

#%%
x1 = liverVol_seg/33.3
x2 = liverVol_pred/33.3

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


#%%
# compare to manual analysis data
pdff_8avg = np.array([2.70,
2.09,
9.08,
3.39,
2.81,
10.93,
5.49,
17.44,
8.39,
1.69,
6.16,
4.58,
10.23,
24.61,
5.10,
4.92,
3.01,
3.19,
2.50,
23.31,
3.23,
24.60,
25.55,
4.52,
2.66,
1.51,
32.09,
3.61,
6.57,
1.99,
5.90,
3.04,
20.70,
10.58,
19.93,
24.43,
6.89,
16.53,
12.36,
3.00])

pdff_wholeliver_pred =np.array([2.964057513,
1.987908925,
9.259303342,
3.178082531,
2.630371365,
10.80626323,
6.197996258,
17.97435003,
8.528459884,
2.051574669,
6.131656788,
3.632764559,
10.70973987,
24.26688688,
2.768575637,
4.760218118,
2.833846485,
3.536821456,
2.639656107,
23.51695483,
3.485845643,
25.58124785,
24.78953581,
4.817887772,
2.54919875,
1.452538097,
31.30676412,
3.20096904,
6.116638772,
2.06190294,
5.200105561,
2.709154986,
19.95650007,
10.02830354,
20.19746119,
24.53125732,
7.198177534,
15.78859361,
11.96562626,
3.240205375])


#%%
# scatter plot and correlation coefficient between volume measurements 
x = pdff_8avg
y = pdff_wholeliver_pred
pearR = np.corrcoef(x,y)[1,0]
print(pearR.shape)
A = np.vstack([x,np.ones(x.shape[0])]).T
m,c = np.linalg.lstsq(A,y)[0]
plt.scatter(x,y, color='red')
plt.plot(x,x*m+c,color='blue',label='r=%3.3f'%(pearR))
plt.legend(loc=2) # legend at upper left
plt.xlabel('mean PDFF mea')
plt.ylabel('Manual liver volumes (cm^3)')

