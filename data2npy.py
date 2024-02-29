from __future__ import print_function

import os
import numpy as np

from skimage.io import imread


#data_path = 'raw/'
trainset_path = 'raw/train/'
testset_path = 'raw/test/'

image_rows = 168
image_cols = 512

def create_train_data():
    #train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(trainset_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        image_path = os.path.join(trainset_path, image_name)
        image = np.array([imread(image_path, as_grey=True)])
        if image_name[-8:-4] =='mask':      
            imgs_mask[i] = image            
        else:
            imgs[i] = image  

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        else: pass
        i += 1
        if i == 527:
            break
        
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    #train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(testset_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])   #숫자 1부터 5,508번까지.
        img = imread(os.path.join(testset_path, image_name), as_grey=True)
        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done!')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_train_data()
    create_test_data()
