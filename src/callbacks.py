#!/usr/bin/env python

# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

from math import ceil
from keras.callbacks import Callback

# user defined libraries
from src.utility import shorten_anatomy_name
from src.metrics import max_index, heatmap_distance
from src.data_generator import data_generator, get_batched_hdf5
from src.utility import generate_image_buffer, shorten_anatomy_name

class HeatmapLocalizationCallback(Callback):
    def __init__(self, data, test_keys, batch_size, input_specs, logdir=None, func_lst=[]):
        # set attributes
        self.data = data
        self.test_keys = test_keys
        self.batch_size = batch_size
        self.logdir = logdir
        self.func_lst = func_lst

        # additional preset attributes
        self.anatomy_lst = input_specs["landmark_order"]
        self.matrix_shape = [
            input_specs["INPUT_Y_SIZE"],
            input_specs["INPUT_X_SIZE"],
            # input_specs["INPUT_Z_SIZE"], #lol 2d HACK
        ]
        self.input_mtx, self.output_mtx = get_batched_hdf5(data, test_keys)
        self.input_mtx = self.input_mtx[:, :, :, 0]

    def tensorboard_images(self, preds, epoch):

        # make nice anatomy names list
        anatomy_names = shorten_anatomy_name(self.anatomy_lst)

        # iterate over anatomies
        for anatomy_indx in range(len(anatomy_names)):

            # get predicted resukts and actual results
            pred_a_indxd = preds[:, :, :, anatomy_indx]
            output_a_indxd = self.output_mtx[:, :, :, anatomy_indx]

            # create image buffer
            img_buff = generate_image_buffer(self.input_mtx, output_a_indxd, pred_a_indxd, 1)

            # decide png
            img = tf.image.decode_png(img_buff.getvalue(), channels=3)
            img = tf.expand_dims(img, 0)

            # add summary
            tf_summ = tf.summary.image(
                # name of image
                "{}-{}".format(
                    anatomy_names[anatomy_indx],
                    self.test_keys[1],
                ),

                # image
                img,
            )

        # run tensorboard
        with tf.Session() as sess:
            # run
            summary = sess.run(tf_summ)

            # write
            writer = tf.summary.FileWriter(self.logdir)
            writer.add_summary(summary)
            writer.close()

    def mean_dist_anatomies(self, preds):
        """
        INPUTS:
            preds:
                the predicton matrix
        EFFECT:
            prints current average test accuracry
        """
        # initialize list
        heatmap_lst = []

        # iterate over anatomies
        for anatomy_indx in range(len(self.anatomy_lst)):
            # get predicted and actual results
            pred_a_indxd = preds[:, :, :, anatomy_indx]
            output_a_indxd = self.output_mtx[:, :, :, anatomy_indx]

            # get max indicies
            pred_indx = max_index(pred_a_indxd, self.matrix_shape)
            actual_indx = max_index(output_a_indxd, self.matrix_shape)

            # get evualation metrics
            mean_dist = np.round(np.mean(heatmap_distance(actual_indx, pred_indx)))
            heatmap_lst.append(mean_dist)

        # bind to dictionary
        heatmap_dict = dict(zip(shorten_anatomy_name(self.anatomy_lst), heatmap_lst))

        # print out
        print("\n")
        print(" - ".join(["{} distance: {}".format(k, v) for k, v in heatmap_dict.items()]))
        print("\n")

    def on_epoch_end(self, epoch, logs={}):
        """
        INPUTS:
            epoch:
                keras parameter for current epoch
            logs:
                keras parameter for log information
        EFFECT:
            runs loaded functions
        """
        # skip if no functions to run
        if len(self.func_lst):
            # get predicted data
            preds = self.model.predict_generator(
                data_generator(
                    self.data,
                    self.test_keys,
                    self.batch_size,
                ),
                steps=ceil(len(self.test_keys)/self.batch_size),
            )

            # run functions
            [getattr(self, func)(preds) for func in self.func_lst]

            self.tensorboard_images(preds, epoch)
