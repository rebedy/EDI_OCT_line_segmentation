import numpy as np
import glob

from keras.preprocessing.image import img_to_array, load_img


class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="data/",
                 train_path="data/train/image", label_path="data/train/label",
                 test_path="data/test", npy_path="npy_data/", img_type="tif"):
        """ """
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.train_path = train_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        print('=' * 30)
        print('Creating training images...')
        print('-' * 30)

        imgs = glob.glob(self.train_path + "/*." + self.img_type)
        labels = glob.glob(self.label_path + '/*.' + self.img_type)
        img_datas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1),
                               dtype=np.uint8)  # 1, 3, 5...none?? it was 1
        label_datas = np.ndarray((len(labels), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        img_names = []
        label_names = []
        for i, img_path in enumerate(imgs):
            img = load_img(img_path, grayscale=True)
            img = img_to_array(img)
            img_datas[i] = img
            img_names.append(img_path.split('\\')[-1])
            if i % 100 == 0:
                print('   Done: {0}/{1} train images'.format(i, len(imgs)))
        print(' ')
        for l, label_path in enumerate(labels):
            label = load_img(label_path, grayscale=True)
            label = img_to_array(label)
            label_datas[l] = label
            label_names.append(label_path.split('\\')[-1])
            if l % 100 == 0:
                print('   Done: {0}/{1} train labels'.format(l, len(labels)))
        print('   loading done')
        np.save(self.npy_path + '/imgs_train.npy', img_datas)
        np.save(self.npy_path + '/imgs_mask_train.npy', label_datas)
        print('saving to .npy files done.')
        return img_names, label_names

    def create_test_data(self):
        print('=' * 30)
        print('Creating test images...')
        print('-' * 30)

        tests = glob.glob(self.test_path + "/*." + self.img_type)
        test_datas = np.ndarray((len(tests), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        test_names = []
        for t, test_path in enumerate(tests):
            test = load_img(test_path, grayscale=True)
            test = img_to_array(test)
            test_datas[t] = test
            test_names.append(test_path.split('\\')[-1])
            if t % 40 == 0:
                print('   Done: {0}/{1} test images'.format(t, len(tests)))
        print('   loading done')
        np.save(self.npy_path + '/imgs_test.npy', test_datas)
        print('saving to .npy files done.')
        return test_names

    def load_train_data(self):
        print('=' * 30)
        print('Load train images...')

        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        # Normalizing
        mean = imgs_train.mean(axis=0)
        std = np.std(imgs_train)
        imgs_train -= mean
        imgs_train /= std
        imgs_train /= 255

        imgs_mask_train /= 255,
        imgs_mask_train = np.where(imgs_mask_train > 0.1, 1, 0)
        # imgs_mask_train[imgs_mask_train > 0.5] = 1
        # imgs_mask_train[imgs_mask_train <= 0.5] = 0

        print('   loading done.')
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('=' * 30)
        print('Load test images...')
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')

        mean = imgs_test.mean(axis=0)
        std = np.std(imgs_test)
        imgs_test -= mean
        # imgs_test /= std
        imgs_test /= 255

        print('   loading done.')
        return imgs_test


if __name__ == "__main__":
    # mydata = dataProcess(160, 512)
    mydata = dataProcess(144, 320)  # (144, 384)
    img_names, label_names = mydata.create_train_data()
    test_names = mydata.create_test_data()
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_test = mydata.load_test_data()
