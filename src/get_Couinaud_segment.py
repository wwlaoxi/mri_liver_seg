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
def generate_couinaud_segments(landmark, liver_seg):
    """
    generate couinaud segments from anatomical landmarks identified on a mask file landmark
    Input:
        landmark -- anatomical landmark for couinaud segments bounardy identification
        liver_seg -- whole liver segmentation 
    
    Output:
        couinaud segmentation mask 
    """
    landmark_location = dict()
    landmark_location['RHV'] = get_centroid_coordinate(landmark,2)
    # for IVC we want it at the level where the RHV, MHV or FL are located to draw the bisection line
    z_slice = np.int(landmark_location['RHV'][2])
    tmp = landmark[:,:,z_slice-1:z_slice+2]
    landmark_location['IVC_RHV'] = get_centroid_coordinate(tmp,1)
    landmark_location['MHV'] = get_centroid_coordinate(landmark,3)
    z_slice = np.int(landmark_location['MHV'][2])
    tmp = landmark[:,:,z_slice-1:z_slice+2]
    landmark_location['IVC_MHV'] = get_centroid_coordinate(tmp,1)
    landmark_location['FL'] = get_centroid_coordinate(landmark,5)
    z_slice = np.int(landmark_location['FL'][2])
    tmp = landmark[:,:,z_slice-1:z_slice+2]
    landmark_location['IVC_FL'] = get_centroid_coordinate(tmp,1)
    landmark_location['LPV'] = get_centroid_coordinate(landmark,4)
    landmark_location['RPV'] = get_centroid_coordinate(landmark,6)


    # obtain planes for dissecting segments
    plane1 = get_bisecting_plane_nl_vec(landmark_location['IVC_FL'],landmark_location['FL'])
    plane2 = get_bisecting_plane_nl_vec(landmark_location['IVC_MHV'],landmark_location['MHV'])
    plane3 = get_bisecting_plane_nl_vec(landmark_location['IVC_RHV'],landmark_location['RHV'])
    
    # get all the coordinates for liver tissues
    xx,yy,zz = np.where(liver_seg ==1)
    Nvoxels = xx.shape[0]
    liver_tissue_coordinates = np.transpose(np.array((xx,yy,zz)),[1,0]) 
    prod1 = np.dot(liver_tissue_coordinates-landmark_location['IVC_FL'],plane1)
    prod2 = np.dot(liver_tissue_coordinates-landmark_location['IVC_MHV'],plane2)
    prod3 = np.dot(liver_tissue_coordinates-landmark_location['IVC_RHV'],plane3)

    #
    segments_label = np.zeros( (Nvoxels,) )
    
    # plane 1 left side is segment 2/3
    segments_label[ np.logical_and(prod1 < 0,zz < landmark_location['LPV'][2])  ] = 2
    segments_label[ np.logical_and(prod1 < 0,zz >= landmark_location['LPV'][2])  ] = 3
    
    # plane 1 right side segment 4a/4b
    segments_label[  np.logical_and( np.logical_and(prod1>=0,prod2<0),zz < landmark_location['LPV'][2] )  ] = 4
    segments_label[  np.logical_and( np.logical_and(prod1>=0,prod2<0),zz >= landmark_location['LPV'][2] )  ] = 9 # use 9 to signifify 4b
    
    # plane 2 right side are right lobes of liver
    segments_label[ np.logical_and( np.logical_and(prod2>=0,prod3<0),zz < landmark_location['RPV'][2]  ) ] = 8
    segments_label[ np.logical_and( np.logical_and(prod2>=0,prod3<0),zz >= landmark_location['RPV'][2]  ) ] = 5
    
    segments_label[ np.logical_and( prod3>=0,zz < landmark_location['RPV'][2]) ] = 7
    segments_label[ np.logical_and( prod3>=0,zz >= landmark_location['RPV'][2]) ] = 6
    segments_mask = voxelCoord_to_mask(liver_tissue_coordinates,segments_label,liver_seg.shape)
    # label from anatomical landmark 7 is segment 1
    segments_mask[ landmark==7 ] = 1
    return segments_mask