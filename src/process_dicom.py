#!/usr/bin/env python

# import libraries
import os
import re
import dicom
import argparse
import glob

import numpy as np

from scipy.misc import imresize

from src.interpolate import linear_interpolate
from operator import attrgetter

#%%
def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


def read_dicom(path):
    """
    INPUTS:
        path:
            a string denoting the path
    OUTPUT:
        list object of dicom files
    """
    # regular expression search for .dcm file
    if re.search(".dcm$", path) is not None:
        return dicom.read_file(path, force=True)

def crop_center(img, factor):
    """
    """
    # determine sizes
    y_size, x_size = img.shape

    # determine min bounds
    new_y = int((y_size//2) - (y_size//factor//2))
    new_x = int((x_size//2) - (x_size//factor//2))

    # determine half values
    half_y, half_x = tuple(int(i//factor) for i in img.shape)

    return img[new_y: new_y + half_y, new_x: new_x + half_x]

def sort_dicom_list(dicom_list):
    """
    INPUTS:
        dicom_list:
            an unsorted list of dicom objects
    OUTPUT:
        sorted list of dicom objects based off of dicom InstanceNumber
    """

    # test that all elements of the list are dicom objects
    if not all([True if type(x) == dicom.dataset.FileDataset else False for x in dicom_list]):
        raise AssertionError("Not all elements are dicom images")
    # sort according slice position so we are ALWAYS going from superior -> inferior
    s_dicom_lst = sorted(dicom_list,key=attrgetter('SliceLocation'),reverse=True)
    
    return s_dicom_lst
    

def process_dicom(path, target_x_size=256, target_y_size=256, target_z_size = 256):
    """
    """
    # initialize rslt_dict
    rslt_dict = {}
    # store files and append path
    dicom_files = glob.glob(os.path.join(path, "*.dcm"))

    # read dicom files
    dicom_lst = [read_dicom(x) for x in dicom_files]
    dicom_lst = [x for x in dicom_lst if x is not None]
    # sort list
    dicom_lst = sort_dicom_list(dicom_lst)

    # return image sizes to result dict
    rslt_dict["slices"] = dicom_lst[-1].InstanceNumber
    rslt_dict["Rows"] = dicom_lst[0].Rows
    rslt_dict["Columns"] = dicom_lst[0].Columns
    
    # alot of time we only want resampling along axial direction for 2D processing
    Nz = np.int( dicom_lst[-1].InstanceNumber - dicom_lst[0].InstanceNumber+1)
    if target_z_size == 0:
        target_z_size = Nz     

    # make a list and cast as 3D matrix
    pxl_lst = [x.pixel_array for x in dicom_lst]

    # downsample to target size
    # need to use floating point mode, otherwise remap to 0 to 255
    pxl_lst = [imresize(x.astype('float32'), (target_y_size, target_x_size),mode='F') for x in pxl_lst]

    pxl_mtx = pxl_lst
    pxl_mtx = linear_interpolate(pxl_lst, target_z_size)

    return pxl_mtx
                     
#%% the following are modified routines to handle multi-echoes MRI series that 
def sort_dicom_list_multiEchoes(dicom_list):
    """
    This function sort first by instance number and then by echo numbers since all six echoes are in the data
    then return a 2-dimensional list that store all images with same echoes into one list
    
    """
    if not all([True if type(x) == dicom.dataset.FileDataset else False for x in dicom_list]):
        raise AssertionError("Not all elements are dicom images")
    
    #s_dicom_lst = sorted(dicom_list,key=attrgetter('InstanceNumber'))
    # sort according to SliceLocation from hight to low so we always go from S->I, instance number doesn't have location information
    s_dicom_lst = sorted(dicom_list,key=attrgetter('InstanceNumber'))
    ss_dicom_lst = sorted(s_dicom_lst,key=attrgetter('EchoNumbers'))
    num_echoes = ss_dicom_lst[-1].EchoNumbers
    dicom_list_groupedby_echoNumber = [None]*num_echoes
    for ii in range(num_echoes):
        tmp_list = []
        for dicomObj in ss_dicom_lst:    
            if dicomObj.EchoNumbers == ii+1:
                tmp_list.append(dicomObj)
        dicom_list_groupedby_echoNumber[ii] = tmp_list
    
    return dicom_list_groupedby_echoNumber 
                     
                     
#%%                     
def process_dicom_multiEcho(path, target_x_size=256, target_y_size=256, target_z_size=0):                    

    result_dict = {}
    # store files and append path
    dicom_files = glob.glob(os.path.join(path, "*.dcm"))
    #dicom_files = [path + "/" + x for x in dicom_files]

    # read dicom files
    dicom_lst = [read_dicom(x) for x in dicom_files]
    dicom_lst = [x for x in dicom_lst if x is not None]
    # sort list
    # this return a 2-dimension list with all dicom image objects within the same 
    # echo number store in the same list
    dicom_lst = sort_dicom_list_multiEchoes(dicom_lst)
    num_echoes = len(dicom_lst)
    print("num of series: "+ str(len(dicom_lst[1])) )
    
    # reports back the first and last instance number of the image sequence
    result_dict['first_instance_number'] = dicom_lst[0][0].InstanceNumber
    result_dict['last_instance_number'] = dicom_lst[-1][-1].InstanceNumber
    # due to sorting by slice location, sometimes instance number can go from big to small
    ##******* NEED TO FIX THIS IN THE FUTURE, IMPORTANT FOR 3D CNN
    Nimg = np.abs( (result_dict['last_instance_number'] - result_dict['first_instance_number'])+1 )
    # return image sizes to result dict
    Nz = np.int( Nimg/num_echoes )
    Ny = np.int( dicom_lst[0][0].Rows )
    Nx = np.int( dicom_lst[0][0].Columns )
    # the following data might not be available due to anonymization
    try:
        result_dict['patientID'] = dicom_lst[0][0].PatientID
        result_dict['AcquisitionDate'] = dicom_lst[0][0].AcquisitionDate
    except:
        pass
    # make a list and cast as 3D matrix for each echo 
    # give the option that don't interpolate along the z-axis if 2-D processing
    if target_z_size == 0:
        target_z_size = Nz    
    
    scale_x = target_x_size/Nx
    scale_y = target_y_size/Ny
    scale_z = target_z_size/Nz
    result_dict['image_scale'] = (scale_x,scale_y,scale_z)
    result_dict['num_echoes'] = num_echoes
    x_sampling = np.float( dicom_lst[0][0].PixelSpacing[0] )
    y_sampling = np.float( dicom_lst[0][0].PixelSpacing[1] )
    z_sampling = np.float( dicom_lst[0][0].SliceThickness )
    result_dict['image_resolution'] = (x_sampling*scale_x,y_sampling*scale_y,z_sampling*scale_z)   
    pxl_mtx = np.zeros( (target_y_size,target_x_size,target_z_size,num_echoes) )
    
    
    for ii in range(num_echoes):
        pxl_lst = [x.pixel_array for x in dicom_lst[ii]]

        # crop file
        #pxl_lst = [crop_center(x, 1.3) for x in pxl_lst]

        # downsample to target size, use floating point mode otherwise remap to 0 to 255
        pxl_lst = [imresize(x.astype('float32'), (target_y_size, target_x_size), mode='F') for x in pxl_lst]
       
        pxl_mtx[:,:,:,ii] = normalize_image( linear_interpolate(pxl_lst, target_z_size) )

    
    result_dict['image_data'] = pxl_mtx
    
    return result_dict  
                     