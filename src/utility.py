#!/usr/bin/env python

# import libraries
import io
import os
import time
import yaml
import shutil
import errno
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import re
from xlrd import open_workbook
from  natsort import natsorted
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
#%%
def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage: 
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()



#%%
def listdir_fullpath(dn):
    
    tmp = [os.path.join(dn,fn) for fn in os.listdir(dn)]
    # sort it at natural order (i.e. 1 2 3 4 ... 10 instead of 1 10 100 etc )
    lst_dir = natsorted(tmp) 
    return lst_dir


#%%
def grab_subdirectory(dn, subdirectory_pattern='\\\\Lipoquant'):
    """
    return the subdirectories that match a specific pattern
    
    """
    dn_list =[]
    subdir_pattern = re.compile(subdirectory_pattern)
    for root, directories, files in os.walk(dn):
        for directory in directories: 
            current_subdirectory =  os.path.join(root, directory) 
            matchobj = subdir_pattern.search(current_subdirectory)
            if matchobj:
                dn_list.append(current_subdirectory)
    return dn_list


#%%
def resample_imgvol(imgvol,output_img_shape):
    # handle data in the format of Nz x Nx x Ny
    tmp = np.transpose(imgvol[:,:,:,0],[1,2,0])
    img_resampled = np.zeros( (tmp.shape[2],output_img_shape[0],output_img_shape[1],1) )
    zoom_x = output_img_shape[0]/tmp.shape[0]
    zoom_y = output_img_shape[1]/tmp.shape[1]
    zoom_z = 1
    tmp2 = zoom(tmp,[zoom_x,zoom_y,1],order=0)     
    img_resampled[:,:,:,0] = np.transpose(tmp2,[2,0,1])
    return img_resampled    
    
    


def write_imgdata_to_nfti(fpath, fname, img, voxel_sampling):
    """ 
    write the output of the downsample images and segmentation mask
    to be view correctly in 3D slicer as .nii format
    Input:
        img: the data is assume to be reorder to Nrow x Ncol x Nz
    
    """
    data_fn = fpath + '/' + fname
    tmp = np.transpose(img,[1,0,2])
    tmp = np.flip( np.flip( np.flip(tmp,axis=2), axis=1), axis=0)
    affine = np.eye(4)
    affine[0,0] = voxel_sampling[0]
    affine[1,1] = voxel_sampling[1]
    affine[2,2] = voxel_sampling[2]
    fnobj = nib.Nifti1Image(tmp,affine)
    nib.nifti1.save(fnobj,data_fn)

    
#%%
def resize_img_stack(img_stack,output_img_shape):
    """
    this helper function resize the first two dimension of the 
    
    
    """
#%%
# copy directory and files into a destination 
def copy_image_series(src, dest):
    try:
        shutil.copytree(src,dest)
    except OSError as e:
        if e.errno==errno.ENOTDIR:
            shutil.copy(src,dest)
        else:
            print('Directory not copied.  Error:%s'%e)



            
#%% helper functio that transform the image volume from ITK-SNAP friendly view
# to the view makes sense in python training
def transform_ITKSNAP_to_training(imgvol):
    tmp = np.flip(np.flip(np.flip(imgvol,axis=0),axis=1),axis=2)
    imgvol2 = np.transpose(tmp,[1,0,2])
    return imgvol2
    

def preprocess_segvol(imgvol,image_output_size):
    """ 
    perform data volume preprocessing 
    imgvol: input image volume
    image_output_size: tupe of output size
    
    """
    tmp   =  imgvol.astype(IMG_DTYPE)
    # determin zoom factor
    zoom_x = image_output_size[0]/imgvol.shape[0]
    zoom_y = image_output_size[1]/imgvol.shape[1]
    zoom_z = image_output_size[2]/imgvol.shape[2]
    tmp =   zoom(tmp,[zoom_x,zoom_y,zoom_z],order=0)     
    # convert back to binary since interpolation might have some values that are not 0 or 1
    tmp[tmp<0.5] = 0.0
    tmp[tmp>=0.5] = 1.0
    return tmp    

    
#%%
def get_FFmap_fn(data_dn):
    pat = re.compile('LQ.*_[0-9]*')
    matchObj = pat.search(data_dn)
    tmp = matchObj.group(0)
    # split it by '_' and get the last element which is seriesID
    tmp2 = re.split('_',tmp)
    len_char = len(tmp2[-1])
    seriesID = 9000 + np.int(tmp2[-1])
    FFmap_pat = 'FF' + tmp[:-1*len_char] + str(seriesID)
    FFmap_fn = re.sub('LQ.*_[0-9]*',FFmap_pat,data_dn)    
    return FFmap_fn


    

    
    