# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:08:06 2017

@author: kang927
"""

%load_ext autoreload
%autoreload 2

import dicom
from  natsort import natsorted
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize,imsave
import nibabel as nib # for reading nifTi data file
from scipy.ndimage.interpolation import zoom
# for postprocessing
from scipy.ndimage import label, binary_erosion, binary_dilation 
from src.dilatedCNN import dilatedCNN, dice_coef_loss, dice_coef, jacc_dist
from keras.models import load_model

# custom build modules
from src.process_dicom import process_dicom_multiEcho, read_dicom
from src.get_Couinaud_segment import get_centroid_coordinate, get_bisecting_plane_nl_vec, voxelCoord_to_mask
from src.utility import write_imgdata_to_nfti, resample_imgvol
#%%
def convert_binary(pred):
    """ 
    the result from u-net prediction is numerical floating pt that is close to 0 or 1
    need to convert to binary
    
    """
    pred = pred.astype(np.float64)
    pred[pred <= 0.5] = 0.0
    pred[pred > 0.5] = 1.0
    return pred

    
def postprocess_cnn(cnn_seg):
    
    nobj = 3
    struct1 = np.ones((nobj,nobj,nobj))
    pred_postp1 = binary_erosion(cnn_seg[:,:,:,0],structure=struct1)
    cnn_seg2 = np.zeros(cnn_seg.shape)
    struct2 = np.ones((3,3,3))
    labels, num_features = label(pred_postp1,struct2)
    feature_size = np.zeros((num_features,))   
    for ii in range(num_features):
        feature_size[ii] = np.sum(labels==(ii+1))
        
    label_liver = np.argmax(feature_size)
    pred_postp = np.zeros( (cnn_seg.shape[0],cnn_seg.shape[1],cnn_seg.shape[2])  )
    pred_postp[labels== (label_liver+1)] = 1
    pred_postp = binary_dilation(pred_postp,struct1)
    cnn_seg2[:,:,:,0]= pred_postp 
    return cnn_seg2
    
#%%
model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_dcnn_10232017'
model = load_model(model_fn, custom_objects={'jacc_dist': jacc_dist, 
                                            'dice_coef': dice_coef})

#%%
echo_num=3
img_dn = 'F:/Block_Ann/Research_Sirlin_Eze_V2 - 6887/LQ_WL_14'
dicom_files = os.listdir(img_dn)
dicom_files = [img_dn + "/" + x for x in dicom_files]
# read dicom files
dicom_lst = [read_dicom(x) for x in dicom_files]
#%%
dicom1 = dicom_lst[0]
dicom2 = dicom_lst[10]
#%%
result = process_dicom_multiEcho(img_dn, target_x_size=192, target_y_size=192,target_z_size=0)
tmp = result['image_data']
img_data = np.transpose(tmp[:,:,:,echo_num-1:echo_num],[2,0,1,3])
img_ds = resample_imgvol(img_data,[128,128,128])
#plt.imshow(img_data[5,:,:])

#%%
pred = model.predict(img_ds,batch_size=8)
pred = convert_binary(pred)
pred_resampled = resample_imgvol(pred,[192,192,192])
pred = postprocess_cnn(pred)  
dsampling = np.array([1.0,1.0,1.0])
data_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/PDFF automated placement/ismrm_figures'
write_liverseg2slicer(img_data,pred_resampled,dsampling,data_fn)



#%% loading in the landmark data
data_dn = 'D:/liverseg_training/mri_training_liverseg/test_cases/'
landmark_fn = data_dn + 'anat_landmark.nii.gz'
landmarkObj = nib.load(landmark_fn)
landmark = landmarkObj.get_data()

    
#%%
# different landmark are ask follow: 1= IVC, 2 = FL, 3= middle hepatic vein, 4 = right hepatic vein, 5 = portal bifu
landmark_location = dict()
landmark_location['IVC'] = get_centroid_coordinate(landmark,1)
landmark_location['FL'] = get_centroid_coordinate(landmark,2)
landmark_location['MHV'] = get_centroid_coordinate(landmark,3)
landmark_location['RHV'] = get_centroid_coordinate(landmark,4)
landmark_location['PVB'] = get_centroid_coordinate(landmark,5)


#%% obtain planes for dissecting segments
plane1 = get_bisecting_plane_nl_vec(landmark_location['IVC'],landmark_location['FL'])
plane2 = get_bisecting_plane_nl_vec(landmark_location['IVC'],landmark_location['MHV'])
plane3 = get_bisecting_plane_nl_vec(landmark_location['IVC'],landmark_location['RHV'])
#%%
# get all the coordinates for liver tissues
mask = np.transpose(pred_resampled[:,:,:,0],[1,2,0])
xx,yy,zz = np.where(mask ==1)
Nvoxels = xx.shape[0]
liver_tissue_coordinates = np.transpose(np.array((xx,yy,zz)),[1,0]) 
prod1 = np.dot(liver_tissue_coordinates-landmark_location['IVC'],plane1)
prod2 = np.dot(liver_tissue_coordinates-landmark_location['IVC'],plane2)
prod3 = np.dot(liver_tissue_coordinates-landmark_location['IVC'],plane3)

#%%
segments_label = np.zeros( (Nvoxels,) )
segments_label[prod1 >= 0] = 1
segments_label[ np.logical_and(prod1<0,prod2>=0)] = 2
segments_label[ np.logical_and(prod2<0,prod3>=0) ] = 3
segments_label[prod3<0] = 4
segments_label[zz<=(26-landmark_location['PVB'][2])] = segments_label[zz<=(26-landmark_location['PVB'][2])] + 4 

segments_mask = voxelCoord_to_mask(liver_tissue_coordinates,segments_label,mask.shape)
#%%
plt.imshow(segments_mask[:,:,23])

#%%
tmp = np.zeros( (segments_mask.shape[2],segments_mask.shape[0],segments_mask.shape[1],1) )
tmp[:,:,:,0] = np.transpose(segments_mask,[2,0,1])
write_liverseg2slicer(img_data,tmp,dsampling,data_fn)

#%%

def read_dicom_series(directory, filepattern = "image_*"):
    """ 
    Reads a DICOM Series files in the given directory. 
    Only filesnames matching filepattern will be considered
    Scale the raw input file to HU from RecaleSlope & RescaleIntercept
    
    """
    
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : "+str(directory))
    print('\tRead Dicom',directory)
    lstFilesDCM = natsorted(glob.glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series',len(lstFilesDCM) )
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # get the space sampling
    dx = np.float(RefDs.PixelSpacing[0])
    dy = np.float(RefDs.PixelSpacing[1])
    dz = np.float(RefDs.SliceThickness)
    dsampling = np.array([dx,dy,dz])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # transform the raw data to HU using Rescale slope and intercept and store it as array 
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom, dsampling


#%%
img_dn = 'F:/test_cases_mri_6E/Arm1_445_MC_010410/Research_Sirlin - 1335/FFLQ_6E_9005/'
img_data, dsampling = read_dicom_series(img_dn,'*.dcm')
fn = 'FFmap.nii'
write_imgdata_to_nfti(img_dn,fn,img_data,dsampling)