"""
#======================================================================================#
    EDI - OCT Dataset stats
# ------------------------------------------------------------------------------- #
        > total # of patients: 221 people
        > total # of slices = 5304 slices
        > size of total dataset: 10,608 (5.20GB)

        > patients for training : 200 people
        > patients for validation : 15 people
        > patients for test : 6 people
        > train + valid data - # of slices : 215 people X 24 slices = 5160 slices
        >> Training set - # of slices : 200 people X 24 slices = 4800 slices
        >> Validation set - # of slices : 15 people X 24 slices = 360 slices

        ---------------
        After crop and all the cleansings...
        * data shape and type : ((144, 320, 3)) -> RGB
        * label shape and type : ((144, 320, 3)) -> bach ground, yellow, green

        ---------------
        total # of marked slices :  5160
        total images with green :  3571
        total images without green :  1589
        train images without green :  1482
        valid images without green :  107

#======================================================================================#
    data_cleansing.py
# ------------------------------------------------------------------------------- #
        < Organizing Filing-System and Data Cleansing >
            * Data Listing and Adjusting to 2-digit (in the Initial_folder)
            * Sorting, cropping, flipping if 'Left' and saving in new directory as rename
            * Extracting label image by np.where
"""


# --------Python in-built modules----------#
import os
import shutil
import random
import numpy as np
import cv2
from PIL import Image
from scipy import misc
# ------------------------------------------#


'''
================================
0) Customized modules defining.
================================
'''


def mkdir_folder_list(src_dir, folder_list, name='', a=None, b=None):
    folder_path = []
    for i, folder in enumerate(folder_list):
        folder_path.append(src_dir + name + folder[a:b] + '/')
        if not os.path.exists(folder_path[i]):
            os.mkdir(folder_path[i])
    return folder_path


def mkdir_folder_path(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        print(folder_path, ' already exists.')
    return folder_path


def mkdir_folder_by_color(seg_path, color_name):
    seg_folder_path = seg_path + color_name + '_segmented/'
    if not os.path.exists(seg_folder_path):
        os.mkdir(seg_folder_path)
    return seg_folder_path


def pati_list_randomizer(src_dir, rand_num, folder_name):
    init_path = os.path.join(src_dir, '_EDI-OCT_dataset_total')

    if not os.path.exists(init_path):
        print("Sorry, there is no folder, %s" % init_path)
        print("Please, check the directory.")
    else:
        patient_folder = os.listdir(init_path)
        patient_path = []
        for i, patient in enumerate(patient_folder):
            patient_path.append(os.path.join(init_path, patient))
        rand_list = random.sample(patient_path, rand_num)
        if os.path.exists(os.path.join(src_dir, folder_name)):
            shutil.rmtree(os.path.join(src_dir, folder_name))
        output_path = mkdir_folder_path(os.path.join(src_dir, folder_name))
        for p, path in enumerate(rand_list):
            if (p + 1) % 15 == 0:
                print('     ...Copying %d th path.' % (p+1))
            # shutil.copytree(path, os.path.join(output_path, path[-10:]))  #!#
            shutil.move(path, os.path.join(output_path, path[-10:]))

    return mkdir_folder_path(os.path.join(src_dir, folder_name))


def make_2digit(src_dir, img, numeric):
    if img[numeric].isnumeric():
        pass
    else:
        new_oct = str(img[0:numeric + 1]) + '0' + str(img[numeric + 1:])
        os.rename(src_dir + str(img), src_dir + new_oct)
        return new_oct


def crop_and_flip(oct_path, dir_path, i, oct_img, patient, label_mode=False):
    pat_id = patient.split('.')[0]
    image = Image.open(oct_path)
    # crop_area = (168, 0, 680, 160)  # (160x512)
    # crop_area = (236, 0, 620, 144)  # (144x384)
    crop_area = (268, 0, 588, 144)  # (144x320)

    if label_mode is False:
        if patient[-1] == 'R':
            cropped = image.crop(crop_area)
            cropped.save(dir_path + i + '_' + pat_id + '_' + oct_img[-6:-4] + '_R' + oct_img[-4:])
        if patient[-1] == 'L':
            cropped = image.crop(crop_area).transpose(Image.FLIP_LEFT_RIGHT)
            cropped.save(dir_path + i + '_' + pat_id + '_' + oct_img[-6:-4] + '_L' + oct_img[-4:])
    else:
        if patient[-1] == 'R':
            cropped = image.crop(crop_area)
            cropped.save(dir_path + i + '_' + pat_id + '_' + oct_img[-7:-4] + '_R' + oct_img[-4:])
        if patient[-1] == 'L':
            cropped = image.crop(crop_area).transpose(Image.FLIP_LEFT_RIGHT)
            cropped.save(dir_path + i + '_' + pat_id + '_' + oct_img[-7:-4] + '_L' + oct_img[-4:])


def data_cleasing(init_mode_dir, mode_dir):
    patient_folder = os.listdir(init_mode_dir)
    patient_path = []

    data_folder = mkdir_folder_path(os.path.join(mode_dir + 'data/'))
    marked_folder = mkdir_folder_path(os.path.join(mode_dir + 'marked/'))
    label_folder = mkdir_folder_path(os.path.join(mode_dir + 'label/'))
    oct_imgs = []  # multi-dimension

    for i, patient in enumerate(patient_folder):
        patient_path.append(os.path.join(init_mode_dir + '/' + patient + '/'))
        oct_lists = os.listdir(patient_path[i])
        oct_imgs.append(oct_lists)

        ############### Crop/Flip/Save each images into suitable folder
        for j, oct_list in enumerate(oct_lists):
            if oct_list[-5] == 'c' or oct_list[-5] == 'C':
                make_2digit(patient_path[i], oct_list, -7)
            else:
                make_2digit(patient_path[i], oct_list, -6)

            oct_path = os.path.join(init_mode_dir + '/' + patient + '/' + oct_imgs[i][j])
            if oct_imgs[i][j][-5] in ('c', 'C'):
                crop_and_flip(oct_path, marked_folder, str(i), oct_imgs[i][j], patient, label_mode=True)
            else:
                crop_and_flip(oct_path, data_folder, str(i), oct_imgs[i][j], patient)

    return data_folder, marked_folder, label_folder


def get_list_path(upper_path, list_name, slash=1):
    get_path = []
    if slash == 0:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name))
    if slash == 1:
        for list_name in list_name:
            get_path.append(os.path.join(upper_path + list_name + '/'))
    return get_path


def extract_label(image):
    ##### Label Extraction for 'RED' and 'BLUE' lines
    r, g, b = cv2.split(image)  # r, g, b = cv2.split(image)
    label_red = np.where(g < b, 1, 0)
    label_blue = np.where(r > b, 1, 0)
    # label_yellow = np.where(r != g, 1, 0)

    ##### Label Extraction for 'YELLOW' and 'GREEN' lines
    yellow_green = image
    for i in range(3):
        yellow_green[:, :, i] = np.where(r != g, yellow_green[:, :, i], 0)
    r, g, b = cv2.split(yellow_green)
    yellow = np.where((r != g) != (g != b), 1, 0)
    green = np.where((r != g) != (g == b), g, 0)
    green = np.where(green != 0, 1, 0)

    ##########################################################################
    ##### Set label as " 5 " channel image by np.where
    red_1in5 = np.where(label_red == 1, 1, 0)
    yellow_2in5 = np.where(yellow == 1, 2, 0)
    green_3in5 = np.where(green == 1, 3, 0)
    blue_4in5 = np.where(label_blue == 1, 4, 0)
    label_5ch = red_1in5 + yellow_2in5 + green_3in5 + blue_4in5

    ##### Set label as " 3 " channel image with yellow and green
    yellow_1in3 = np.where(yellow == 1, 1, 0)
    green_2in3 = np.where(green == 1, 2, 0)
    yellow_green_3ch = yellow_1in3 + green_2in3

    ##### Set label as " 1 " channel image for each colors
    red = np.where(label_red == 1, 1, 0)
    yellow = np.where(yellow == 1, 1, 0)
    green = np.where(green == 1, 1, 0)
    blue = np.where(label_blue == 1, 1, 0)
    yellow_green_1ch = yellow + green

    return label_5ch, red, yellow, green, blue, yellow_green_3ch, yellow_green_1ch


'''
================================
1) Dataset cleansing part
================================
'''

def data_cleansing(src_dir, dataset_dir):

    print('Start random sampling dataset.')
    init_train_dir = pati_list_randomizer(src_dir, 200, '_EDI-OCT_train')
    init_valid_dir = pati_list_randomizer(src_dir, 15, '_EDI-OCT_valid')
    # init_test_dir  = pati_list_randomizer(src_dir,  22, 'EDI-OCT_test' )
    print('\n     Total train set length : ', len(os.listdir(init_train_dir)))
    print('     Total valid set length : ', len(os.listdir(init_valid_dir)))
    print('     ...Dataset randomly separating done!')

    print('Start train dataset cleansing.')
    if os.path.exists(os.path.join(dataset_dir, 'training/')):
        shutil.rmtree(os.path.join(dataset_dir, 'training/'))
    train_dir = mkdir_folder_path(os.path.join(dataset_dir, 'training/'))
    train_data_dir, train_marked_dir, train_label_dir = data_cleasing(init_train_dir, train_dir)
    print('Start valid dataset cleansing.')
    if os.path.exists(os.path.join(dataset_dir, 'validation/')):
        shutil.rmtree(os.path.join(dataset_dir, 'validation/'))
    valid_dir = mkdir_folder_path(os.path.join(dataset_dir, 'validation/'))
    # test_dir  = mkdir_folder_path(os.path.join(total_dir + 'test/'))
    valid_data_dir, valid_marked_dir, valid_label_dir = data_cleasing(init_valid_dir, valid_dir)
    print('     ...Dataset cleansing done!')

    ##### list of images that are marked - train
    print('Train label image extraction start.')
    train_marked_oct = os.listdir(train_marked_dir)
    train_marked_path = get_list_path(train_marked_dir, train_marked_oct, 0)
    for i, tmpath in enumerate(train_marked_path):
        t_image = cv2.imread(tmpath)
        t_label_5ch, t_red, t_yellow, t_green, t_blue, t_yellow_green_3ch, t_yellow_green_1ch = extract_label(t_image)
        # label_5ch, red, yellow, green, blue, yellow_green_3ch, yellow_green_1ch
        misc.imsave(train_label_dir + train_marked_oct[i], t_yellow_green_3ch)
    print("      Train label extraction done. Please check 'train_label_dir'")
    print("      ", train_label_dir)
    print('Validation label image extraction start.')

    ##### list of images that are marked - valid
    valid_marked_oct = os.listdir(valid_marked_dir)
    valid_marked_path = get_list_path(valid_marked_dir, valid_marked_oct, 0)
    for i, vmpath in enumerate(valid_marked_path):
        v_image = cv2.imread(vmpath)
        v_label_5ch, v_red, v_yellow, v_green, v_blue, v_yellow_green_3ch, v_yellow_green_1ch = extract_label(v_image)
        misc.imsave(valid_label_dir + valid_marked_oct[i], v_yellow_green_3ch)
    print("      Validation label extraction done. Please check 'valid_label_dir'")
    print("      ", valid_label_dir, '\n')
    print('Label extraction done!')


if __name__ == '__main__':
    
    src_dir = 'DATASET_STORAGE/EDIOCT_dataset_total/'
    if len(os.listdir(os.path.join(src_dir, 'EDI-OCT_dataset'))) == 0:
        input('EDI-OCT_dataset folder is empty. Please restore dataset.')
    dataset_dir = os.path.join(src_dir, 'EDI-OCT_dataset')
    print('Total dataset length : ', len(os.listdir(dataset_dir)), '\n')

    data_cleansing(src_dir, dataset_dir)
