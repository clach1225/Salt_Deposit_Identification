import random
import pandas as pd
import os

from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#from PIL import Image, ImageFilter

import keras.backend as KerasBackend

from skimage.transform import resize

import time


from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
import keras
# import cv2

import tensorflow as tf
import keras.utils as np_utils


import tqdm

from keras.preprocessing import image
from keras.layers import Lambda
from keras.layers import Concatenate

from keras.callbacks import EarlyStopping
from tqdm import tqdm_notebook as tqdm_NB
import os
import shutil



import skimage.io as io
import skimage.transform as trans



from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import re

from IPython.core.display import display, HTML


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")



def data_importing(path_train = 'C:/Users/buckf/Documents/Practicum_2/Salt_Deposit_Identification-master/data/train',
                   path_test = 'C:/Users/buckf/Documents/Practicum_2/Salt_Deposit_Identification-master/data/test'):


    train_ids = next(os.walk(path_train+"/images"))[2]
    test_ids = next(os.walk(path_test+"/images"))[2]

    im_height = 128
    im_width = 128
    im_chan = 1

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm_NB(enumerate(train_ids), total=len(train_ids)):
        path = path_train
        img = image.load_img(path + '/images/' + id_)
        x = image.img_to_array(img)[:,:,1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        X_train[n] = x
        mask = image.img_to_array(image.load_img(path + '/masks/' + id_))[:,:,1]
        Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

    print('Done!')

    return X_train, Y_train

def data_importing_test(path_train = 'C:/Users/buckf/Documents/Practicum_2/Salt_Deposit_Identification-master/data/train',
                        path_test = 'C:/Users/buckf/Documents/Practicum_2/Salt_Deposit_Identification-master/data/test'):
    train_ids = next(os.walk(path_train+"/images"))[2]
    test_ids = next(os.walk(path_test+"/images"))[2]

    im_height = 128
    im_width = 128
    im_chan = 1

    # Get and resize test images
    X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm_NB(enumerate(test_ids), total=len(test_ids)):
        path = path_test
        img = image.load_img(path + '/images/' + id_)
        x = image.img_to_array(img)[:,:,1]
        sizes_test.append([x.shape[0], x.shape[1]])
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        X_test[n] = x

    print('Done!')

    return X_test, sizes_test, test_ids
