import numpy as np

def linear_interpolate(pxl_lst, new_cols):
    """
    INPUTS:
        pxl_lst:
            list of sorted 2D images
        new_cols:
            the int for the target number of z slices
    OUTPUT:
        a matrix with apprioprate number of slices
    """
    # make matrix of values
    vals = np.stack(pxl_lst, axis=-1)

    # define cols
    old_cols = vals.shape[2]

    # define seq of values
    old = np.linspace(0, 1, old_cols)
    new = np.linspace(0, 1, new_cols)

    # determine alignment
    old_mtx = np.vstack([old] * new_cols)
    new_mtx = np.vstack([new] * old_cols).T

    # determine index
    idx_low = (old_mtx < new_mtx).sum(axis=1)
    idx_high = (old_mtx <= new_mtx).sum(axis=1) - 1
    idx_equal = (old_mtx == new_mtx).sum(axis=1)

    # determine width
    width = 1 / (old_cols - 1)

    # determine differance
    diff_high = abs(new - old[idx_low])/width
    diff_low = abs(new - old[idx_high])/width

    # calculate averages
    new_vals = (vals[:, :, idx_low] * diff_low) + (vals[:, :, idx_high] * diff_high)

    # determine where val are equal
    idx_replace_new = np.where(idx_equal)
    idx_replace_old = np.where(np.isin(old, new[idx_replace_new]))

    # replace these values
    new_vals[:, :, idx_replace_new] = vals[:, :, idx_replace_old]

    return new_vals

def nearest_neighbor_interpolate(dicom_lst, target_z_size, how="max"):
    """
    """
    old = np.linspace(0, 1, 6)
    new = np.linspace(0, 1, 5)

    old_mtx = np.vstack([old] * new.shape[0])
    new_mtx = np.vstack([new] * old.shape[0]).T

    if how == "max":
        min_idx = np.argmin(np.fliplr(abs(new_mtx - old_mtx)), axis=1)
    elif how == "min":
        min_idx = np.argmin(abs(new_mtx - old_mtx), axis=1)
    else:
        raise AssertionError("How must be either 'min' or 'max'")

    # subset, and cast to matrix, and return
    vals = np.stack([x.pixel_array for x in dicom_lst[min_idx]], axis=-1)
    return vals
