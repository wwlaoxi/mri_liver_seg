#!/ussr/bin/env python

# import libraries
import h5py
import numpy as np

from math import ceil, floor

def load_data(data_path):
    """
    INPUTS:
        data_path:
            the path for the HD5F file
    OUTPUT:
        the data
    """

    # read in file
    return h5py.File(data_path + "/data.hdf5", "r")

def validation_data(data, train_prop):
    """
    INPUTS:
        data:
            the h5py data object
        train_prop:
            the proprotion of data to use in training data set
    OUTPUT:
        tuple:
            0: training value keys
            1: testing value keys
    """
    # determine values
    val_arry = np.array(list(data.keys()))
    np.random.shuffle(val_arry)

    # determine train and test vals
    train_idx = ceil(train_prop * val_arry.shape[0])

    train_vals = val_arry[:train_idx]
    test_vals = val_arry[train_idx:]

    return train_vals, test_vals

def get_batched_hdf5(data, curr_keys):
    """"
    INPUTS:
        data:
            the h5py data object
        curr_values:
            the values to use
    OUTPUT:
        tuple:
            0: stacked X_values
            1: stacked Y_value
    """

    # return lists of input and output
    x_lst = [data[x + "/input"] for x in curr_keys]
    y_lst = [data[x + "/output"] for x in curr_keys]

    # return matrix
    x_lst = [np.expand_dims(x[()].astype('float32'), axis=-1) for x in x_lst]
    y_lst = [x[()].astype('float32') for x in y_lst]

    # HACK for 2D
    idx_lst = [np.unravel_index(x[:, :, :, 1].argmax(), [128] * 3)[2] for x in y_lst]

    x_lst = [x[:, :, y] for x, y in zip(x_lst, idx_lst)]
    y_lst = [x[:, :, y] for x, y in zip(y_lst, idx_lst)]

    x_rtn = np.stack(x_lst, axis=0)
    y_rtn = np.stack(y_lst, axis=0)

    return x_rtn, y_rtn

def data_generator(data, values, batch_size):
    """"
    INPUTS:
        data:
            the h5py data object
        values:
            the training values to use
        batch_size:
            the number of datasets used
   OUTPUT:
        tuple:
            0: input into network
            1: output into network
    """

    # make batch sized indicies
    min_cuts = floor(len(values)/ batch_size)
    slices = np.arange(0, min_cuts*batch_size).reshape(min_cuts, batch_size).tolist()

    # if there's a remainder, append at end of list
    if len(values) % batch_size:
        slices.append(np.arange(min_cuts*batch_size, len(values)).tolist())

    # loop through values
    while 1:
        for curr_idx in slices:
            yield get_batched_hdf5(data, values[curr_idx])
