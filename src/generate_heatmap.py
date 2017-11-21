#!/usr/bin/env python

# import libraries
import numpy as np
from scipy.stats import multivariate_normal

def generate_heatmap(xyz_location, xyz_limits, sigma):
    """
    INPUTS:
        xyz_location:
            tuple of x, y, and z coordinates for centering heatmap[
        xyz_limits:
            tuple of x, y, and z coordinates for the map
        sigma:
            standard deviation
    """
    # assemble bins
    bins = np.indices([xyz_limits[0], xyz_limits[1], xyz_limits[2]])
    pos = np.stack(bins, axis=-1)

    # define covariance matrix
    cov = [[sigma, 0, 0], [0, sigma, 0], [0, 0, sigma]]

    # create prob density function
    var = multivariate_normal(mean=xyz_location, cov=cov)

    # normalize to 1
    rtn_mtx = var.pdf(pos)

    min_val = rtn_mtx.min()
    max_val = rtn_mtx.max()

    rtn_mtx = (rtn_mtx - min_val)/(max_val - min_val)

    return rtn_mtx
