# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:15:59 2017

dilation convolution to process data 

@author: kang927
"""

import os

import glob

import numpy as np


from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, merge,Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dense, Dropout, Activation
from keras.layers import MaxPooling2D, UpSampling2D,Flatten,concatenate 
from keras import regularizers 
from keras.layers.normalization import BatchNormalization as bn
from keras import initializers
from keras import backend as K


# loss and accuracy metric for segmentation


def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum( y_true_f * y_pred_f )

    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)

    
def jacc_dist(y_true, y_pred):
    
    smooth=1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    jacc_dist = 1 - (intersection + smooth) / ( K.sum(y_true_f) + K.sum(y_pred_f) + smooth - intersection )
    return jacc_dist



seed=0
N_featuremaps = 64
#sdddev = 0.01

def dilatedCNN(input_shape, l2_lambda=0.0001,DropP=0.1):
    """
    Implement the multi-scale context aggregation by dilated convolution by Yu et. al. in ICLR 2016
    
    Argument: 
        input_shape: input data shape
        l2_lambda: l2 normalization parameter
        dropP: drop_out rate during training
    
    """
    
    
    inputs = Input(input_shape)
    
    # add a gaussian layer to add noise for data augmentation
    #noisy_inputs = noise.GaussianNoise(stddev)(inputs)
    
    conv1 = Conv2D( N_featuremaps, (3, 3), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(inputs)
    
    conv1 = bn()(conv1)
    conv1 = Dropout(DropP)(conv1)
    
    
    conv1 = Conv2D( N_featuremaps, (3, 3), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv1)
    
    conv1 = bn()(conv1)
    conv1 = Dropout(DropP)(conv1)
    
    conv2 = Conv2D( N_featuremaps, (3, 3), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv1)
    
    conv2 = bn()(conv2)
    conv2 = Dropout(DropP)(conv2)
    
    conv3 = Conv2D( N_featuremaps, (3, 3), dilation_rate =2, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                       kernel_initializer=initializers.he_normal() )(conv2)

    conv3 = bn()(conv3)
    conv3 = Dropout(DropP)(conv3)
    
    conv4 = Conv2D( N_featuremaps, (3, 3), dilation_rate =4, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv3)

    conv4 = bn()(conv4)
    conv4 = Dropout(DropP)(conv4)
    
    conv5 = Conv2D( N_featuremaps, (3, 3), dilation_rate =8, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv4)
    

    conv5 = bn()(conv5)
    conv5 = Dropout(DropP)(conv5)
    
    conv6 = Conv2D( N_featuremaps, (3, 3), dilation_rate =16, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv5)

    conv6 = bn()(conv6)
    conv6 = Dropout(DropP)(conv6)
    
    
    conv7 = Conv2D( N_featuremaps, (3, 3), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv6)
    
    
    # insert non-dilation cnn layers to see if help with segmentation not scattering
    conv7 = Conv2D( N_featuremaps, (3, 3), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv7)
    
    conv7 = bn()(conv7)
    conv7 = Dropout(DropP)(conv7)
    
    
    conv7 = Conv2D( N_featuremaps, (3, 3), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv7)
    

    conv7 = bn()(conv7)
    conv7 = Dropout(DropP)(conv7)
    
    conv8 = Conv2D( 192, (1, 1), dilation_rate =1, activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda),
                    kernel_initializer=initializers.he_normal() )(conv7)
    
    conv8 = bn()(conv8)
    conv8 = Dropout(DropP)(conv8)
    
    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv8)
    
    model = Model(inputs=inputs, outputs=conv9)
    return model

if __name__ == "__main__":

    model = dilatedCNN(input_shape=(256, 256, 1))
    print(model.summary())