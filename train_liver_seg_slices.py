# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:59:09 2017

@author: kang927
"""


"""
Created on Sat Aug 26 11:36:21 2017

@author: kang927

script to setup training data and data augmentation generator 
for training of liver segmentation

"""
%load_ext autoreload
%autoreload 2

import numpy as np
import os
import glob
# for visual displlay of vollume
from matplotlib import pyplot as plt
from IPython import display
# use for natural order sorting
from natsort import natsorted
# random shuffle input images for better training
from random import shuffle
import random

# keras
from keras.optimizers import Adam
from keras import initializers 
from keras.preprocessing.image import ImageDataGenerator

# custom function 
from src.dilatedCNN import dilatedCNN, dice_coef_loss, dice_coef, jacc_dist
from src.dilatedresidualCNN import ResnetBuilder
from src.utility import listdir_fullpath, imshow
from src.train import get_subset, hist_match,batch_read, batch_read_multiecho, batch_read_2echo

#%% here we will set the training and test data set 
img_dir = 'D:/liverseg_training/mri_training_liverseg/training/images'
seg_dir = 'D:/liverseg_training/mri_training_liverseg/training/segmentations'
list_img = listdir_fullpath(img_dir)
list_seg = listdir_fullpath(seg_dir)

#%%
# setup training and test data

# only grab the out-of-phase echoes
echo_list = [0,1,2,3,4,5]
img_list_train = [ii for ii in range(143)] + [ii for ii in range(174,241)]
img_list_test = [ii for ii in range(143,173)] 
train_img_list,train_seg_list = get_subset(img_dir,seg_dir,echo_list,img_list_train)
test_img_list, test_seg_list = get_subset(img_dir,seg_dir,echo_list,img_list_test)

# obtaining the testing set 
test_list = (test_img_list,test_seg_list)

# for the training set we want to randomly shuffle the input data for better training
tmp = list(zip(train_img_list,train_seg_list))
shuffle(tmp)
train_img_list,train_seg_list = zip(*tmp)
train_list = (train_img_list,train_seg_list)


#%%
# utilize keras own imagedatagenerator for data augmentation
data_gen_args = dict(rotation_range=20.,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     zoom_range=0.1,
                     shear_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

images,masks, tmp = batch_read(train_list,10,0)
#images,masks, tmp = batch_read_multiecho(train_list,10,0,[0,1,2,3,4,5])
# Provide the same seed and keyword arguments to the fit and flow methods
#%%
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)
#%%
seed = np.random.randint(0,10000)
# these are the actual image and mask generators
img_gen = image_datagen.flow(images[:,:,:,0:1],seed=seed)
#img_gen2 = image_datagen.flow(images[:,:,:,1:2],seed=seed)
mask_gen = mask_datagen.flow(masks[:,:,:,0:1],seed=seed)

#tmp1 = next(img_gen)
#tmp2 = next(mask_gen) 
#tmp3 = next(img_gen2)

#%% 
def batch_generator(data_list,
                    batch_size,
                    image_generator=None,
                    mask_generator=None):
    """ 
    generator that iterate through batch of volume data and generate
    augmented 2D images for training
    
    Argument:
        img_seg_generator: keras ImageDataGenerator object for images and segmentations 
        data_list: a list contain the directory of data volume in .npy format
        batch_vol: how many volume data to load at each iteration
        batch_size: number of images to generate per iteration
    """
    Nslices = len(data_list[0]) # number of data slices to loop through
    ii=0
    
    while 1:
        #print("using data slice #" + str(ii) + "\n")
        # load batches of img and segmentation        
        #img_train, mask_train, ii = batch_read_multiecho(data_list,batch_size,ii,[0,1,2,3,4,5])
        img_train, mask_train, ii = batch_read(data_list,batch_size,ii)
       # reset if we reach the end of all files
        if ii%Nslices ==0:
            #print("reseting ii to 0\n")
            ii=0 # if we loop through all data vol, repeat again
        
    
        # perform data augmentation
        x_train = np.zeros(img_train.shape)
        num_echo = img_train.shape[3]
        # perform histogram matching if needed
        for img_ii in range(batch_size):
            hist_match_ii = np.random.randint(0,batch_size-1)
            # we allow probability 1/2 time to perform histogram equilization and other 1/2 times will be original images
            if np.random.uniform() <0.2:
                for echo_ii in range(num_echo):
                    img_train[img_ii,:,:,echo_ii] = hist_match(img_train[img_ii,:,:,echo_ii],img_train[hist_match_ii,:,:,echo_ii])

        if image_generator:
            seed = np.random.randint(0,10000)
            # since image_gen only takes in data 1,3,4 at axis 3, we need to initially create a data-generator
            for echo_num in range(num_echo):
                image_gen = image_generator.flow(img_train[:,:,:,echo_num:echo_num+1],seed=seed)
                x_train[:,:,:,echo_num:echo_num+1]  = next(image_gen)

            mask_gen = mask_generator.flow(mask_train, seed=seed)
            y_train = next(mask_gen)
        else:
            x_train = img_train
            y_train = mask_train
        
        yield x_train, y_train


#%%

# testing for the generator
ct = 0
train_generator = batch_generator(train_list,20,image_datagen,mask_datagen)
test_generator = batch_generator(test_list,20)
for x_test, y_test in train_generator:
#    print(x_test.shape)
    imshow(x_test[5,:,:,0],y_test[5,:,:,0],y_test[5,:,:,0])
    ct = ct+1
    if ct>10:
        break
#tmp = next(test_generator)
#print(hasattr(tmp, '__len__'))



#%% initialize the model and training parameters
batch_size = 32
n_train = len(train_img_list)
n_test = len(test_img_list)
print("using "+ str(n_train)+" for training and using "+ str(n_test)+" for testing")


#%%
nx = 224
ny = 224
n_channels = 1

# make the training and test data generator
train_generator = batch_generator(train_list,batch_size,image_datagen,mask_datagen)
test_generator = batch_generator(test_list,batch_size)



#%%
#from keras.models import load_model
#model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/mri_liver_seg/liverseg_mri6E_nodropout_dcnn_10232017'
#model_fn = 'C:/Users/kang927/Documents/deep_learning_liverseg/LiverSeg/liverseg_unet_slab1_liveronly_axial_dataAugment_nodropout_101517'
#model = load_model(model_fn, custom_objects={'jacc_dist': jacc_dist, 
#                                            'dice_coef': dice_coef})

#%%
#weights = model2.get_weights()
#model = dilatedCNN( input_shape=(nx,ny,n_channels),l2_lambda=0.001,DropP=0.0)
model = ResnetBuilder.build_resnet_18(input_shape=(n_channels,224,224),num_outputs=1)
#model.set_weights(weights)
#model=UNet( (224,224,n_channels), out_ch=1, start_ch=16, depth=3, inc_rate=2., activation='relu', 
#		 dropout=0.0, batchnorm=True, maxpool=True, upconv=True, residual=False)

#%% setup learning parameters and metric for optimization 
model.compile(optimizer=Adam(lr=1e-5), loss=jacc_dist, metrics=[dice_coef])


#%% start training
num_epochs=20
n_per_epoch = np.round(n_train/batch_size)
n_test_steps = np.round(n_test/batch_size)
hist1 = model.fit_generator(train_generator,steps_per_epoch=n_per_epoch,epochs=num_epochs,
                               validation_data=test_generator,
                               validation_steps=n_test,
                               verbose=1)

#%%
model_fn ='liverseg_mri6E_nodropout_drsn_augment_histeq_individualEcho_12227017'
model.save(model_fn)

#%% save the hist object that store the optimization
import pickle
hist_fn = model_fn + '_optimization_history'
file_hist = open(hist_fn,'wb')
pickle.dump(hist1.history,file_hist)
file_hist = open(hist_fn,'rb')
hist2 = pickle.load(file_hist)
file_hist.close()
#%%
test_generator2 = batch_generator(test_list, n_test)
x_test,y_test = next( test_generator2 )
pred = model.predict(x_test,batch_size=8)
#%%
diceScore = np.zeros((pred.shape[0],)) 
for ii in range(pred.shape[0]):
    diceScore[ii] = dice_coef_py(pred[ii,:,:,0],y_test[ii,:,:,0])
    
plt.hist(diceScore,bins=30)
#%%

indbadcase = np.where(diceScore<0.6)
Nv = len(indbadcase[0])
for ii in range(0,Nv,2):
    s = indbadcase[0][ii]
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(10,8))
    fig1 = ax.imshow(x_test[s,:,:,0],cmap='gray')
    fig1 = ax.imshow(pred[s,:,:,0], cmap=plt.cm.inferno, alpha=0.3)
    
#%%
# summarize history for accuracy
history = hist1
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model DICE score')
plt.ylabel('DICE score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%%
# summarize history for loss
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%

                   

