# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:07:46 2017

@author: kang927
"""
import numpy as np
import os, glob
import re

#%%
def batch_read(data_list,batch_size,ii):
    """ 
    load up batch of data and prepare it for training 
    Argument:
        data_list: tuple of img and seg file list
        batch_size: number of batch data to load
        ii: index indicates starts reading from iith file in data_list
    
    """
    ct = 0
    Nslice = len(data_list[0])
    
    # take care of the case that the next batch of data run over the end of all slices
    if ii+batch_size >= Nslice:
        index_end = Nslice
    else:
        index_end = ii+batch_size
    
    # load the first image to get the dimension of the image
    tmp= np.load(data_list[0][ii])
    nx = tmp.shape[0]
    ny = tmp.shape[1]
    img = np.zeros( (batch_size,nx,ny,1) )
    seg = np.zeros( (batch_size,nx,ny,1) )
              
    for s in range(ii,index_end,1):
        img[ct,:,:,0] = np.load(data_list[0][s])
        seg[ct,:,:,0] = np.load(data_list[1][s])
        ct = ct+1
    # repackage the shape to make it fit keras training data 
    #print("the img size is" + str(img.shape) + "\n")
    return img, seg, s+1

#%%
def batch_read_2echo(data_list,batch_size,ii):
    ct = 0
    Nslice = len(data_list[0])
    
    # take care of the case that the next batch of data run over the end of all slices
    if ii+batch_size >= Nslice:
        index_end = Nslice
    else:
        index_end = ii+batch_size
    
    # load the first image to get the dimension of the image
    tmp= np.load(data_list[0][ii])
    nx = tmp.shape[0]
    ny = tmp.shape[1]
    img = np.zeros( (batch_size,nx,ny,2) )
    seg = np.zeros( (batch_size,nx,ny,1) )

    for s in range(ii,index_end,1):
        pat = re.compile('echo[0-9]')
        matchobj = pat.search(data_list[0][s])
        echo_num = np.int( matchobj.group(0)[4] ) + 1  
        ip_fn = re.sub("echo[0-9]",'echo'+str(echo_num),data_list[0][s])
        #print(data_list[0][s])
        #print(ip_fn)
        # check whether it is a slice of the liver
        check_liver = np.sum( np.load(data_list[1][s]) )
        
        if check_liver >0:
            img[ct,:,:,0] = np.load(data_list[0][s])
            img[ct,:,:,1] = np.load(ip_fn)
            seg[ct,:,:,0] = np.load(data_list[1][s])
            ct = ct+1
    # repackage the shape to make it fit keras training data 
    #print("the img size is" + str(img.shape) + "\n")
    return img, seg, s+1
    
    
#%%
def batch_read_multiecho(data_list,batch_size,ii, echo_list=[1]):
    ct = 0
    Nslice = len(data_list[0])
    
    # take care of the case that the next batch of data run over the end of all slices
    if ii+batch_size >= Nslice:
        index_end = Nslice
    else:
        index_end = ii+batch_size
    
    # load the first image to get the dimension of the image
    tmp= np.load(data_list[0][ii])
    nx = tmp.shape[0]
    ny = tmp.shape[1]
    num_echo = len(echo_list)
    img = np.zeros( (batch_size,nx,ny,num_echo) )
    seg = np.zeros( (batch_size,nx,ny,1) )

    for s in range(ii,index_end,1):
        for echo_num in echo_list:
            ip_fn = re.sub("echo[0-9]",'echo'+str(echo_num),data_list[0][s])
            img[ct,:,:,echo_num] = np.load(ip_fn)
            #print(data_list[0][s])
            #print(ip_fn)        
        seg[ct,:,:,0] = np.load(data_list[1][s])
        ct = ct+1
    # repackage the shape to make it fit keras training data 
    #print("the img size is" + str(img.shape) + "\n")
    return img, seg, s+1

#%%
def get_subset(img_dir,seg_dir,echo_num_list, image_num_list):
    """
    helper function to get subset of images by the echo number and image index

    """
    subset_img_list =[]
    subset_seg_list = []    

    for img_num in image_num_list:
        for echo_num in echo_num_list:            
            filepattern = '*image' + str(img_num) + '_echo' + str(echo_num)+"*"
            filepattern2 = '*seg' + str(img_num) + '_echo' + str(echo_num)+"*"
            tmp_list = glob.glob(os.path.join(img_dir, filepattern))
            tmp2_list = glob.glob(os.path.join(seg_dir, filepattern2))
            subset_img_list = subset_img_list + tmp_list
            subset_seg_list = subset_seg_list + tmp2_list
            
    # now we will match the segmentation file for the corresponding list in the image list
    #for tmp_file  in tmp_list:
    return subset_img_list, subset_seg_list
    # after we get all subset of echoes 

#%%
###########################%% Data augmentation routines ###########################################################
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


#%%

    