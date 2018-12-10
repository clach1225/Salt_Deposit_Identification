import Augmentor

import os
import shutil
import numpy as np
import sys
from tqdm import tqdm_notebook as tqdm_NB
from keras.preprocessing import image
from skimage.transform import resize


def aug(train_image_path = r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\images",
        train_mask_path = r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\masks",
        path_train = r'C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train',
        num_images_augmented = 1000):

    p = Augmentor.Pipeline(train_image_path)
    # Point to a directory containing ground truth data.
    # Images with the same file names will be added as ground truth data
    # and augmented in parallel to the original data.
    p.ground_truth(train_mask_path)
    # Add operations to the pipeline as normal:
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)

    #This function allows you to select how many images you would like to augment.
    p.sample(num_images_augmented)

    #This takes all the augmentated images and masks and creates places them
    # in a two seperate folders into the train directory.

    srcpath = r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\images\output"
    destpath = r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train"

    for root, subFolders, files in os.walk(srcpath):
        for file in files:
            subFolder = os.path.join(destpath, file[:8])
            if not os.path.isdir(subFolder):
                os.makedirs(subFolder)
            shutil.move(os.path.join(root, file), subFolder)

    #Renames files in train directory for ease.

    #Augmented_Masks:
    os.rename(src = r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\_groundt", dst=r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\aug_masks")

    #Augmented_Images:
    os.rename(src = r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\images_o", dst=r"C:\Users\buckf\Documents\Practicum_2\Salt_Deposit_Identification-master\data\train\aug_images")

    aug_train_images_ids = next(os.walk(path_train+"/aug_images"))[2]
    aug_train_masks_ids = next(os.walk(path_train+"/aug_masks"))[2]



    im_height = 128
    im_width = 128
    im_chan = 1

    from keras.preprocessing import image
    
    # Get and resize train images and masks
    Aug_X_train = np.zeros((len(aug_train_images_ids), im_height, im_width, im_chan), dtype=np.uint8)
    Aug_Y_train = np.zeros((len(aug_train_images_ids), im_height, im_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm_NB(enumerate(aug_train_images_ids), total=len(aug_train_images_ids)):
        path = path_train
        img = image.load_img(path + '/aug_images/' + id_)
        x = image.img_to_array(img)[:,:,1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        Aug_X_train[n] = x

    for n, id_ in tqdm_NB(enumerate(aug_train_masks_ids), total=len(aug_train_masks_ids)):
        mask = image.img_to_array(image.load_img(path + '/aug_masks/' + id_))[:,:,1]
        Aug_Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

    print('Done!')

    return Aug_X_train, Aug_Y_train
