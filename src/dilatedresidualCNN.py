from __future__ import division
"""
Created on Thu Oct 19 20:59:27 2017

@author: kang927
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:15:51 2017
Implemented dildated residual network for liver segmentation 

@author: kang927
"""

import six
#import numpy as np
#from keras.optimizers import Adam, SGD
#from keras.models import Sequential, Model

#from keras.layers import Input, Dense, Convolution2D, merge,Conv3D, noise
#from keras.layers import BatchNormalization, Dense, Dropout, Activation
#from keras.layers import MaxPooling3D, UpSampling3D,Flatten,concatenate 
#from keras import regularizers 
#from keras.layers.normalization import BatchNormalization as bn
#from keras import initializers



from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Concatenate,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    UpSampling2D,
    Conv2DTranspose
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K





def preprocess_input(X):

    return np.asarray([((x - np.mean(x)) / np.std(x)) for x in X])


def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)
    
    y_pred_f = K.flatten(y_pred)

    # y_pred_f is not binary so need to threshold values <=0.5 to 0 and >0.5 to 1
    comparison = K.less_equal( y_pred_f, K.constant( 0.5 ) )    
    y_pred_f = K.tf.where ( comparison, K.zeros_like(y_pred_f), y_pred_f)
    comparison = K.greater( y_pred_f, K.constant( 0.5 ) )
    y_pred_f = K.tf.where (comparison, K.ones_like(y_pred_f), y_pred_f)
    
    intersection = K.sum( y_true_f * y_pred_f )

    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)

    
def jacc_dist(y_true, y_pred):
    
    smooth=1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f is not binary
        
    intersection = K.sum(y_true_f * y_pred_f)
    jacc_dist = 1 - (intersection + smooth) / ( K.sum(y_true_f) + K.sum(y_pred_f) + smooth - intersection )
    return jacc_dist

#%%
l2_lambda = 1e-3    
do =0.0
#%% residual network implementation
def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(l2_lambda))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params["dilation_rate"]
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(l2_lambda))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate = dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(l2_lambda))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, dilation, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
                # we don't use downsampling, but rather dilation convolution to get long range interaction 
                init_strides = (1, 1)
                input = block_function(filters=filters, init_strides=init_strides,dilation=dilation,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
                
                input = Dropout(do)(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), dilation=1,is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           dilation_rate=1,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(l2_lambda))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides, dilation_rate=dilation)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3), dilation_rate=dilation)(conv1)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, dilations):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv0 = _conv_bn_relu(filters=16, kernel_size=(7, 7), strides=(2, 2))(input)
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
        conv1 = _conv_bn_relu(filters=16, kernel_size=(3, 3), strides=(1, 1))(conv0)
        conv1 = _conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(2, 2))(conv1)
        
        block = conv1
        filters = 32
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, 
                                    filters=filters, 
                                    repetitions=r, 
                                    dilation=dilations[i], 
                                    is_first_layer=(i == 0))(block)
            block = Dropout(do)(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)
        # upsampling (4x sub-sampling during decoding)
        up1 = Conv2DTranspose(filters=32,
                              kernel_size=(3,3),
                              strides=(2,2),
                              padding="same",
                              activation='relu',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(l2_lambda))(block)
        tmp1 = Concatenate()([conv0,up1])
        up2 = Conv2DTranspose(filters=16,
                              kernel_size=(7,7),
                              strides=(2,2),
                              padding="same",
                              activation='relu',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(l2_lambda))(tmp1)
        
        # final dense prediction block 
        #block_shape = K.int_shape(block)
        #pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
        #                         strides=(1, 1))(block)
        #flatten1 = Flatten()(pool2)
        #dense = Dense(units=num_outputs, kernel_initializer="he_normal",
        #              activation="softmax")(flatten1)
        output = Conv2D(1, (1, 1), activation='sigmoid')(up2)
        
        
        
        model = Model(inputs=input, outputs=output)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2],[2,4,8,4])

if __name__ == "__main__":
    num_outputs = 10
    model = ResnetBuilder.build_resnet_18(input_shape=(3,128,128),num_outputs=num_outputs)
    print(model.summary())       
    
 
