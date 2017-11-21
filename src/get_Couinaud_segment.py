"""
Created on Thu Nov  2 21:31:19 2017

@author: kang927
"""
import numpy as np

# helper functions that identify the eight Couniaud segments of the liver
#%%
def get_centroid_coordinate(landmark,label):
    x,y,z = np.where(landmark==label)
    coordinates = np.array([np.mean(x),np.mean(y),np.mean(z)])
    return coordinates
    
    
def get_bisecting_plane_nl_vec(vector1, vector2):
    """ 
    get the normal vector to the plane perpendicular to axial-imaging plane 
    and also contain the vector connecting vector1 & vector2
    
    """    
    vector3 = np.copy(vector2)
    # the plane is perpendicular to x-y/axial plane, so any z coordinates are fine
    vector3[2] = vector3[2] - 1
    in_plane_vec1 = vector1 - vector2 
    in_plane_vec2 = vector1 - vector3 
    nl_vec = np.cross(in_plane_vec1,in_plane_vec2)
    nl_vec = nl_vec/np.linalg.norm(nl_vec,2)
    return nl_vec

#%%
def voxelCoord_to_mask(voxel_coordinates,voxel_label,mask_shape):
    """
    convert voxel coordinates into a label mask
    voxel_coordinates: Nvoxels x 3
    voxel_label: Nvoxels x 1
    
    """
    mask = np.zeros(mask_shape)
    Nvoxels = voxel_coordinates.shape[0]
    for ii in range(Nvoxels):
        mask[voxel_coordinates[ii,0],voxel_coordinates[ii,1],voxel_coordinates[ii,2]] = voxel_label[ii]
    
    return mask
    
    

#%%

