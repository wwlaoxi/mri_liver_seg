#!/usr/bin/env python

# import libraries

from keras.models import Model, Input
from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error
from keras.initializers import RandomNormal, Zeros
from keras.initializers import Zeros, RandomNormal
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, \
    UpSampling2D, Activation, AveragePooling2D, merge, Flatten, Dense

def get_2D_u_net(downsize_filter_factor, input_specs, settings):
    """
    INPUTS:
        downsize_filter_factor:
            the amount to downsample conv output layers
        input_specs:
            the dictionary that contains the input specifications for the current data
    OUTPUT:
        compilled model where inputs are a tensor of (n, X_dim, Y_dim, 1)
    """
    top_kernel_size = settings["TOP_KERNEL_SIZE"]
    bottom_kernel_size = settings["BOTTOM_KERNEL_SIZE"]

    input_shape = (
        input_specs["INPUT_Y_SIZE"],
        input_specs["INPUT_X_SIZE"],
        1,
    )

    # define inputs
    inputs = Input(input_shape)

    # define first convolutional layer
    conv1 = Convolution2D(int(32/downsize_filter_factor), top_kernel_size, padding="same", activation="relu", )(inputs)
    conv1 = Convolution2D(int(32/downsize_filter_factor), top_kernel_size, padding="same", activation="relu", )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # define second convolutional layer
    conv2 = Convolution2D(int(64/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(pool1)
    conv2 = Convolution2D(int(64/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # define third convolutional layer
    conv3 = Convolution2D(int(128/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(pool2)
    conv3 = Convolution2D(int(128/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # define fourth convolutional layer
    conv4 = Convolution2D(int(256/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(pool3)
    conv4 = Convolution2D(int(256/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # define fifth convolutional layer
    conv5 = Convolution2D(int(512/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(pool4)
    conv5 = Convolution2D(int(512/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv5)

    # define sixth convolutional layer
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merged6 = concatenate([up6, conv4], axis=-1)
    conv6 = Convolution2D(int(256/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(merged6)
    conv6 = Convolution2D(int(256/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv6)

    # define seventh convolutional layer
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merged7 = concatenate([up7, conv3], axis=-1)
    conv7 = Convolution2D(int(128/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(merged7)
    conv7 = Convolution2D(int(128/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv7)

    # define eighth convolutional layer
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merged8 = concatenate([up8, conv2], axis=-1)
    conv8 = Convolution2D(int(64/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(merged8)
    conv8 = Convolution2D(int(64/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv8)

    # define ninth convolutional layer
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merged9 = concatenate([up9, conv1], axis=-1)
    conv9 = Convolution2D(int(32/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(merged9)
    conv9 = Convolution2D(int(32/downsize_filter_factor), bottom_kernel_size, padding="same", activation="relu", )(conv9)

    # define final output layer
    conv10 = Convolution2D(input_specs["channels"], 1, kernel_initializer=RandomNormal(mean=0., stddev=0.01))(conv9)

    # define model and compile
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(
        loss=mean_squared_error,
        optimizer=SGD(
            lr=float(settings["LEARNING_RATE"]),
            decay=float(settings["WEIGHT_DECAY"]),
            momentum=float(settings["MOMENTUM"]),
        ),
    )

    return model

if __name__ == '__main__':
    main()
