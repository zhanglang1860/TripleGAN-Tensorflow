from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import h5py
from util import log
from sklearn.model_selection import KFold
import random
import tensorflow as tf


def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    flip = random.getrandbits(1)
    if flip:
        image = image[:, ::-1, :]
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
              init_x: init_x + init_shape[0],
              init_y: init_y + init_shape[1],
              :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images




def __init__(self, path, ids, name='default',
             max_examples=None, is_train=True,hdf5FileName=None):
    self._ids = list(ids)
    self.name = name
    self.is_train = is_train

    if max_examples is not None:
        self._ids = self._ids[:max_examples]

    filename = hdf5FileName


    file = os.path.join(path, filename)
    log.info("Reading %s ...", file)

    self.data = h5py.File(file, 'r')
    log.info("Reading Done: %s", file)
    self._batch_counter = 0



def get_data(id, all_hdf5_data):
    # preprocessing and data augmentation
    # id= int(id)
    images = all_hdf5_data[id]['image'].value
    l = all_hdf5_data[id]['label'].value.astype(np.float32)


    return images, l





# def start_new_epoch(self):
#     self._batch_counter = 0
#     # if self.shuffle_every_epoch:
#     #     images, labels = self.shuffle_images_and_labels(
#     #         self.images, self.labels)
#     # else:
#     images, labels = self.images, self.labels
#     # if self.augmentation:
#     #     images = augment_all_images(images, pad=4)
#     self.epoch_images = images
#     self.epoch_labels = labels


def all_images_labels(self):
    id_slice = self._ids
    images_slice = []
    labels_slice = []

    for each_id in id_slice:
        each_images_slice,each_labels_slice= self.get_data(each_id)
        shape_list=each_images_slice.shape
        each_images_slice=np.reshape(each_images_slice, (shape_list[0],shape_list[1],shape_list[2],1))
        images_slice.append(each_images_slice)
        labels_slice.append(each_labels_slice)

    labels_slice = np.array(labels_slice, dtype=np.float32)
    images_slice = np.array(images_slice, dtype=np.float32)
    # if images_slice.shape[0] != batch_size:
    #     self.start_new_epoch()
    #     return self.next_batch(batch_size)
    # else:
    return images_slice, labels_slice





def next_batch(self, batch_size):
    start = self._batch_counter * batch_size
    end = min((self._batch_counter + 1) * batch_size,len(self.ids))
    self._batch_counter += 1
    id_slice = self._ids[start: end]
    images_slice = []
    labels_slice = []

    for each_id in id_slice:
        each_images_slice,each_labels_slice= self.get_data(each_id)
        shape_list=each_images_slice.shape
        each_images_slice=np.reshape(each_images_slice, (shape_list[0],shape_list[1],shape_list[2],1))
        images_slice.append(each_images_slice)
        labels_slice.append(each_labels_slice)

    labels_slice = np.array(labels_slice, dtype=np.float32)
    images_slice = np.array(images_slice, dtype=np.float32)
    # if images_slice.shape[0] != batch_size:
    #     self.start_new_epoch()
    #     return self.next_batch(batch_size)
    # else:
    return images_slice, labels_slice


def create_default_splits(path, hdf5FileName,idFileName,cross_validation_number):
    filename = hdf5FileName

    file = os.path.join(path, filename)
    log.info("Reading %s ...", file)

    all_hdf5_data = h5py.File(file, 'r')
    log.info("Reading Done: %s", file)


    dataset_train, dataset_test = all_ids(path,idFileName,cross_validation_number)
    return dataset_train, dataset_test, all_hdf5_data



def all_ids(path,idFileName,cross_validation_number):
    id_filename = idFileName
    id_txt = os.path.join(path, id_filename)
    with open(id_txt, 'r') as fp:
        ids = [s.strip() for s in fp.readlines() if s]
    # rs = np.random.RandomState(123)
    # rs.shuffle(ids)
    # create training/testing splits
    # train_ratio = 0.8
    # train_ids = ids[:int(train_ratio*len(ids))]
    # test_ids = ids[int(train_ratio*len(ids)):]
    train_ids =[]
    test_ids = []
    dataset_train = []
    dataset_test = []
    for i in range(cross_validation_number):
        train_ids.append([])
        test_ids.append([])
        dataset_train.append([])
        dataset_test.append([])

    kf = KFold(n_splits=cross_validation_number)
    i = 0


    for cross_train_ids, cross_test_ids in kf.split(ids):
        train_ids_one_fold = []
        test_ids_one_fold = []
        for train_index in range(len(cross_train_ids)):
            train_ids_one_fold.append(ids[cross_train_ids[train_index]])

        for test_index in range(len(cross_test_ids)):
            test_ids_one_fold.append(ids[cross_test_ids[test_index]])


        train_ids[i]=train_ids_one_fold
        test_ids[i]=test_ids_one_fold



        dataset_train[i] = list(train_ids[i])
        dataset_test[i] = list(test_ids[i])
        i=i+1

    return dataset_train, dataset_test


def create_default_splits8020(path, hdf5FileName_train,hdf5FileName_test,hdf5FileName_val, idFileName_train,idFileName_test,idFileName_val, num_less_label_data,class_num):

    file_train = os.path.join(path, hdf5FileName_train)
    log.info("Reading %s ...", file_train)

    all_hdf5_data_train = h5py.File(file_train, 'r')
    log.info("Reading Done: %s", file_train)



    file_test = os.path.join(path, hdf5FileName_test)
    log.info("Reading %s ...", file_test)

    all_hdf5_data_test = h5py.File(file_test, 'r')
    log.info("Reading Done: %s", file_test)

    file_val = os.path.join(path, hdf5FileName_val)
    log.info("Reading %s ...", file_val)

    all_hdf5_data_val = h5py.File(file_val, 'r')
    log.info("Reading Done: %s", file_val)





    dataset_train_unlabelled = all_ids8020(path,idFileName_train)
    dataset_test = all_ids8020(path, idFileName_test)
    dataset_val = all_ids8020(path, idFileName_val)

    if num_less_label_data==0:
        dataset_train_labelled=dataset_train_unlabelled
    else:
        dataset_train_labelled = []
        count_each_class=num_less_label_data//int(class_num)
        count_class_0 = 0
        count_class_1 = 0
        count_class_2 = 0
        if int(class_num) == 2:
            for index in range(len(dataset_train_unlabelled)):
                img, label = get_data(dataset_train_unlabelled[index], all_hdf5_data_train)
                if count_class_0 < count_each_class and int(label[0]) == 1:
                    count_class_0 = count_class_0 + 1
                elif count_class_1 < count_each_class and int(label[1]) == 1:
                    count_class_1 = count_class_1 + 1
                else:
                    dataset_train_labelled.append(dataset_train_unlabelled[index])
        else:
            for index in range(len(dataset_train_unlabelled)):
                img, label = get_data(dataset_train_unlabelled[index], all_hdf5_data_train)
                if count_class_0 < count_each_class and int(label[0]) == 1:
                    count_class_0 = count_class_0 + 1
                elif count_class_1 < count_each_class and int(label[1]) == 1:
                    count_class_1 = count_class_1 + 1
                elif count_class_2 < count_each_class and int(label[2]) == 1:
                    count_class_2 = count_class_2 + 1
                else:
                    dataset_train_labelled.append(dataset_train_unlabelled[index])




    return dataset_train_unlabelled, dataset_test, all_hdf5_data_train,all_hdf5_data_test,dataset_train_labelled, dataset_val,all_hdf5_data_val



def all_ids8020(path,idFileName):
    id_filename = idFileName
    id_txt = os.path.join(path, id_filename)
    with open(id_txt, 'r') as fp:
        ids = [s.strip() for s in fp.readlines() if s]
    # rs = np.random.RandomState(123)
    # rs.shuffle(ids)
    # create training/testing splits
    # train_ratio = 0.8
    # train_ids = ids[:int(train_ratio*len(ids))]
    # test_ids = ids[int(train_ratio*len(ids)):]


    return ids



