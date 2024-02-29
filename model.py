import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import numpy as np

from keras.models import *
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import *
from keras.callbacks import TensorBoard, ModelCheckpoint#, LearningRateScheduler
from keras.preprocessing.image import array_to_img
from keras import backend as K
from keras.utils import to_categorical

import data_processing as data


class myUnet(object):
                        # img_rows=144, img_cols=320, / img_rows=160, img_cols=512,
    def __init__(self, img_rows=144, img_cols=320, npy_path="npy_data/", pred_path="preds/",
                 log_path = "log_hist/", img_type="tif"):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.npy_path = npy_path
        self.pred_path = pred_path
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.img_type = img_type
        self.smooth = 1.

    def load_data(self):
        mydata = data.dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        # imgs_mask_train = to_categorical(imgs_mask_train, 5)
        # img_names, mask_name = mydata.create_train_data()
        test_names = mydata.create_test_data()
        return imgs_train, imgs_mask_train, imgs_test, test_names

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (K.sum(y_true_f * y_true_f)
                                                    + K.sum(y_pred_f * y_pred_f) + self.smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)


    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols,1))
                                                               # kernel_initializer='he_normal' 'glorot_uniform'
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        print("conv4 shape:", conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print("conv4 shape:", conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        print("drop4 shape:", drop4.shape)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print("pool4 shape:", pool4.shape)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        print("conv5 shape:", conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        print("conv5 shape:", conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print("drop5 shape:", drop5.shape)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        print("up6 shape:", up6.shape)
        concatenate6 = Concatenate(axis=3)([drop4, up6])
        print("concatenate6 shape:", concatenate6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate6)
        print("conv6 shape:", conv6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print("conv6 shape:", conv6.shape)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        print("up7 shape:", up7.shape)
        concatenate7 = Concatenate(axis=3)([conv3, up7])
        print("concatenate7 shape:", concatenate7.shape)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate7)
        print("conv7 shape:", conv7.shape)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print("conv7 shape:", conv7.shape)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        print("up8 shape:", up8.shape)
        concatenate8 = Concatenate(axis=3)([conv2, up8])
        print("concatenate8 shape:", concatenate8.shape)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate8)
        print("conv8 shape:", conv8.shape)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        print("conv8 shape:", conv8.shape)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        print("up9 shape:", up9.shape)
        concatenate9 = Concatenate(axis=3)([conv1, up9])
        print("concatenate9 shape:", concatenate9.shape)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate9)
        print("conv9 shape:", conv9.shape)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:", conv9.shape)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:", conv9.shape)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        #conv10 = Conv2D(5, 1, activation='sigmoid')(conv9)
        print("conv10 shape:", conv10.shape)

        model = Model(inputs=inputs, outputs=conv10)
        #parallel_model = multi_gpu_model(model, gpus=8)
        #parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model.compile(optimizer=Adam(lr=0.0001), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        #model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0),
                                                 # loss= self.dice_coef_loss, metrics=[self.dice_coef])
        #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
            # 이 문제는 절대 categorial_crossentropy를 쓰지 않는다. classification 문제가 아니므로.
        return model


    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()[0:3]
        #imgs_mask_train = to_categorical(imgs_mask_train, num_classes=5)
        print("loading data done")

        model = self.get_unet()
        print("got unet")


        checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        #model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=300, verbose=1,
        #          validation_split=0.1, shuffle=True, callbacks=[checkpoint])

        tensorboard = TensorBoard(log_dir=self.log_path, histogram_freq=0, write_graph=True, write_images=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=200, verbose=1,  shuffle=True,
                  validation_split=0.1,  callbacks=[tensorboard])  #validation_data=(imgs_train, imgs_mask_train),


        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

        np.save(os.path.join(self.npy_path, 'imgs_mask_test.npy'), imgs_mask_test)


    def save_img(self):
        print('=' * 35)
        print("Array to image")
        imgs = np.load(os.path.join(self.npy_path, 'imgs_mask_test.npy'))
        test_names = self.load_data()[3]
        print('-' * 35)
        print("saving to image...")

        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save(os.path.join(self.pred_path+test_names[i][0:-3] + self.img_type))
        print("   saving to image done.")
        print('=' * 35)


if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
