#!/usr/bin/env python

# import libraries
import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import ceil
from functools import reduce, partial
from multiprocessing import Pool

# user defined libraries
from src.metrics import average_attentuation, heatmap_distance, max_index
from src.data_generator import get_batched_hdf5, data_generator

def write_result_image(curr_anatomy, test_keys, input_mtx, output_a_indxd, pred_a_indxd, path, curr_indx):
    """
    INPUTS:
        curr_anatomy:
            the string of the current anatomy to create
        test_keys:
            the test key list to iterate over
        input_mtx:
            the current input matrix
        output_a_indxd:
            the constructed heatmap output matrix for the current anatomy
        pred_a_indxd:
            the heatmap prediciton matrix for the current anatomy
        path:
            the base path to use
        curr_indx:
            the index for the current test key to use
    EFFECT:
        writes the image
    """
    # make write path (assumes this will be a valid path)
    write_path = "{}/{}".format(path, test_keys[curr_indx])

    # get image arrays
    curr_input = input_mtx[curr_indx, :, :,]
    curr_output = output_a_indxd[curr_indx, :, :,]
    curr_pred = pred_a_indxd[curr_indx, :, :,]

    # set picutres
    fig, axarr = plt.subplots(2)

    axarr[0].imshow(curr_input, cmap="gray")
    axarr[0].imshow(curr_output, cmap=plt.cm.inferno, alpha=0.45)

    axarr[1].imshow(curr_pred, cmap=plt.cm.inferno)

    # write to file
    fig.savefig(write_path + "/" + curr_anatomy + ".png")
    plt.close('all')

def write_multi_output_img(curr_anatomy, test_keys, input_mtx, output_a_indxd, pred_a_indxd, path, settings):
    """
    INPUTS:
        curr_anatomy:
            the string of the current anatomy to create
        test_keys:
            the test key list to iterate over
        input_mtx:
            the current input matrix
        output_a_indxd:
            the constructed heatmap output matrix for the current anatomy
        pred_a_indxd:
            the heatmap prediciton matrix for the current anatomy
        path:
            the base path to use
        settings:
            the setting dict
    EFFECT:
        writes the multi-study output image
    """
    # make multi-study image
    multi_output_indx = np.arange(0, settings['RESULT_STUDIES'])

    # make write path (assumes this will be a valid path)
    write_path = "{}/multi-output".format(path)

    # set picutres
    fig, axarr = plt.subplots(
        nrows=2,
        ncols=len(multi_output_indx),
        sharex=True,
        sharey=True,
    )

    for i in multi_output_indx:
        # target
        axarr[0, i].imshow(input_mtx[i, :, :,], cmap="gray")
        axarr[0, i].imshow(output_a_indxd[i, :, :,], cmap=plt.cm.inferno, alpha=0.45)

        # prediction
        axarr[1, i].imshow(pred_a_indxd[i, :, :, ], cmap=plt.cm.inferno)

    # write to file
    fig.savefig(write_path + "/" + curr_anatomy + ".png")
    plt.close('all')

def generate_result_images(curr_anatomy, test_keys, input_mtx, output_a_indxd, pred_a_indxd, path, settings):
    """
    INPUTS:
        curr_anatomy:
            the string of the current anatomy to create
        test_keys:
            the test key list to iterate over
        input_mtx:
            the current input matrix
        output_a_indxd:
            the constructed heatmap output matrix for the current anatomy
        pred_a_indxd:
            the heatmap prediciton matrix for the current anatomy
        path:
            the base path to use
    EFFECT:
        creates an image file
    """
    # pool multiprocessing
    p = Pool()

    # make a list to index keys
    test_key_indx = list(range(len(test_keys)))

    # make partial
    func = partial(
        write_result_image, # main function
        curr_anatomy,
        test_keys,
        input_mtx,
        output_a_indxd,
        pred_a_indxd,
        path,
    )

    # map function
    p.map(func, test_key_indx)

    # clean up
    p.close()
    p.join()

    # write multi-study output
    write_multi_output_img(
        curr_anatomy,
        test_keys,
        input_mtx,
        output_a_indxd,
        pred_a_indxd,
        path,
        settings,
    )

def evaluate_model(test_keys, data, loaded_model, settings, input_specs, cmd_args, iter_path):
    """
    INPUTS:
        test_keys:
            list of test keys
        data:
            the dhf5 object
        loaded_model:
            the model with either loaded weights or trained weights
        settings:
            the setting dict
        input_specs:
            the dictionary that contains the input specifications for the current data
    OUTPUT:
        pandas dataframe of the distances
    EFFECT:
        creates images
    """
    # construct directories
    [os.makedirs(iter_path + "/" + x) for x in test_keys]
    os.makedirs(iter_path + "/multi-output")

    # get actual results
    input_mtx, output_mtx = get_batched_hdf5(data, test_keys)
    input_mtx = input_mtx[:, :, :, 0]

    # HACK: only for a single channel on 2D for correct layer
    a_indx = np.where(np.array(input_specs["landmark_order"]) == 'Pulmonary Artery Bifurcation')[0].tolist()[0]

    # get predicted data
    preds = loaded_model.predict_generator(
        data_generator(
            data,
            test_keys,
            settings["BATCH_SIZE"],
        ),
        steps=ceil(len(test_keys)/settings["BATCH_SIZE"]),
    )

    # determine matrix shape
    matrix_shape = [
        input_specs["INPUT_Y_SIZE"],
        input_specs["INPUT_X_SIZE"],
        # input_specs["INPUT_Z_SIZE"], # 2d HACK
    ]

    # initialize matrix shape
    rslt_df = pd.DataFrame()

    # iterate over anatomies
    for anatomy_indx in range(len(input_specs["landmark_order"])):

        # get predicted resukts and actual results
        pred_a_indxd = preds[:, :, :, anatomy_indx]
        output_a_indxd = output_mtx[:, :, :, anatomy_indx]

        # get max indicies
        pred_indx = max_index(pred_a_indxd, matrix_shape)
        actual_indx = max_index(output_a_indxd, matrix_shape)

        # get evualation metrics
        heat_dist = heatmap_distance(actual_indx, pred_indx)

        atten_actual = average_attentuation(input_mtx, actual_indx, settings['EVAL_KERNEL_SIZE'])
        atten_pred = average_attentuation(input_mtx, pred_indx, settings['EVAL_KERNEL_SIZE'])
        atten_diff = atten_actual - atten_pred

        # determine current anatomy
        curr_anatomy = input_specs["landmark_order"][anatomy_indx]

        # create a dataframe and add to return data frame
        tmp_df = pd.DataFrame({
            "heatmap_dist": heat_dist,
            "attentuation_diff": atten_diff,
            "test_keys": test_keys,
            "anatomy": curr_anatomy,
        })
        rslt_df = rslt_df.append(tmp_df, ignore_index=True)

        # write images
        if cmd_args.heat_map:
            generate_result_images(
                curr_anatomy=curr_anatomy,
                test_keys=test_keys,
                input_mtx=input_mtx,
                output_a_indxd=output_a_indxd,
                pred_a_indxd=pred_a_indxd,
                path=iter_path,
                settings=settings,
            )

    # rename vars
    rslt_df = rslt_df.rename(columns = {
        "variable": "anatomy",
        "value": "distance",
    })

    return rslt_df
