#!/usr/bin/env python

# import libraries
import os

import numpy as np
import pandas as pd

from math import ceil
from operator import mul
from functools import reduce
from scipy.spatial import distance
from collections import OrderedDict
from keras.callbacks import TensorBoard, ModelCheckpoint

# user defined libraries
from src.evaluate_model import evaluate_model
from src.data_generator import get_batched_hdf5, data_generator
from src.callbacks import HeatmapLocalizationCallback

class CrossValidation:
    def __init__(self, model_func, cmd_args, settings, input_specs, write_path):
        """
        INPUTS:
            model_func:
                the model function to return the current model
            cmd_args:
                the command line arguements
            settings:
                the ai settings file
            input_specs:
                the input specifications from data processing
            write_path:
                the path to write out to
        EFFECT:
            initializes the cross validation to run KxK cross fold validation
        """

        # return self data
        self.model_func = model_func
        self.cmd_args = cmd_args
        self.settings = settings
        self.input_specs = input_specs
        self.write_path = write_path

    def _initialize_folds(self, data, k):
        """
        INPUTS:
            data:
                the hdf5 file path to load
            k:
                the number K cross folds
        EFFECT:
            sets the self.folds_dict
        """
        # determine values
        val_arry = np.sort(np.array(list(data.keys())))
        np.random.shuffle(val_arry)

        # determine correct index
        slice_idx = np.tile(np.arange(0, k), ceil(len(val_arry)/k))[:len(val_arry)]

        # intiialize dict
        self.folds_dict = {x: [] for x in range(0, k)}

        # map keys to lists
        [self.folds_dict[v].append(k) for k, v in OrderedDict(zip(val_arry, slice_idx)).items()]

    def _return_cv_iter(self, k):
        """
        returns the iterable for train and testing keys
        """
        self.curr_iter = 0

        while self.curr_iter < k:
            # make a last of train keys
            train_lsts = [v for k, v in self.folds_dict.items() if k != self.curr_iter]
            train = np.array([item for sublist in train_lsts for item in sublist])

            # make a list of test keys
            test = np.array(self.folds_dict[self.curr_iter])

            # incriment
            self.curr_iter += 1

            yield train, test

    def run_cv(self, data, k):
        """
        INPUTS:
            data:
                the hdf5 file path to load
            k:
                the number K cross folds
        EFFECT:
            trains models, tests models, and writes evaluation information
        OUTPUT:
            pandas dataframe of evaluation metrics
        """

        # initialize callbacks
        tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)

        # initialize data splits
        self._initialize_folds(data, k)

        # initialize output dict
        rslt_df = pd.DataFrame()

        for curr_train, curr_test in self._return_cv_iter(k):
            # reinitialize callback
            cb = HeatmapLocalizationCallback(
                data=data,
                test_keys=curr_test,
                batch_size=self.settings['BATCH_SIZE'],
                input_specs=self.input_specs,
                logdir='./logs',
                func_lst=[
                    "mean_dist_anatomies",
                ],
            )

            # label information
            print("Iteration: {}".format(self.curr_iter))

            curr_iter_name = "iter_{}".format(self.curr_iter)
            iter_path = self.write_path + "/" + curr_iter_name
            os.makedirs(iter_path)

            # reinitialize model
            temp_model = self.model_func()

            # fit model
            temp_model.fit_generator(
                generator=data_generator(data, curr_train, self.settings["BATCH_SIZE"]),
                steps_per_epoch=len(curr_train)/self.settings["BATCH_SIZE"],
                epochs=self.settings["EPOCHS"],
                verbose=1,
                callbacks=[
                    cb,
                ],
            )

            # evualate model
            cv_iter = evaluate_model(
                test_keys=curr_test,
                data=data,
                loaded_model=temp_model,
                settings=self.settings,
                input_specs=self.input_specs,
                cmd_args=self.cmd_args,
                iter_path=iter_path
            )

            # name iteration and add to dataframe
            cv_iter["iter"] = curr_iter_name
            rslt_df = rslt_df.append(cv_iter, ignore_index=True)

            # determine if needed to break loop
            if not self.cmd_args.cross_validation:
                break

        return rslt_df
