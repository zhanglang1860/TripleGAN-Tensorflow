from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from config import argparser

import os
import time

from six.moves import xrange
from pprint import pprint
import h5py
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from util import log
from config import argparser
import datasets.hdf5_loader as dataset
from datetime import timedelta
import random
from operator import itemgetter
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from sklearn.metrics import classification_report
from data_providers.utils import get_data_provider_by_path
from models.dense_net_3d import TripleGAN3D


def construct_train_dir(config):
    all_results_file_name = []
    all_train_dir = []



    temp = config.hdf5FileNametrain.split('.')
    hyper_parameter_all_folds = 'all8020_{}_lr_g_{}_d_{}_num_less_label_data_{}_batchSize{}_maxIteration{}'.format(
        temp[0], config.learning_rate_g, config.learning_rate_d,
        config.num_less_label_data, config.batch_size, config.max_training_steps
    )

    config.prefix = 'TripleGan_depth{}_growthRate{}_reduce{}_model_type{}_keepPro{}_d_loss_version{}'.format(
        config.depth, config.growth_rate, config.reduction,
        config.model_type, config.keep_prob,config.d_loss_version
    )

    train_dir = './train_dir/%s-%s' % (
        hyper_parameter_all_folds, config.prefix,
    )

    if tf.gfile.Exists(train_dir):
        log.infov("Train Dir exists")
    else:
        os.makedirs(train_dir)

    log.infov("Train Dir: %s", train_dir)
    result_file_name = hyper_parameter_all_folds + config.prefix + '-' + time.strftime("%Y%m%d-%H%M%S")

    all_train_dir.append(train_dir)
    all_results_file_name.append(result_file_name)

    return all_train_dir, all_results_file_name


def average_gradients(tower_grads):

  average_grads = []
  for grad_and_vars in zip(*tower_grads):

    grads = []
    for g, _ in grad_and_vars:

      expanded_g = tf.expand_dims(g, 0)

      grads.append(expanded_g)

    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads






def calculateConfusionMatrix(each_result_file_name,class_labels,train_dir):
    df = pd.read_csv(train_dir + '/GANresults/' + each_result_file_name + '.csv')

    df.head()

    cr = classification_report(df.actual_label.values, df.model_GAN.values, target_names=class_labels)
    cm = np.array2string(confusion_matrix(df.actual_label.values, df.model_GAN.values))

    accuracy = accuracy_score(df.actual_label.values, df.model_GAN.values)

    print('Accuracy GAN: %.3f' % (accuracy))

    return accuracy, cr,cm





def main(argv=None):  # pylint: disable=unused-argument

    config = argparser(is_train=True)
    all_train_dir, all_result_file_name = construct_train_dir(config)

    if config.hdf5FileNametrain == 'train_MRIdata_3_AD_MCI_Normal.hdf5':
        dataset_path = os.path.join(r"./datasets/mri/3_AD_MCI_Normal/")
    elif config.hdf5FileNametrain == 'train_MRIdata_2_AD_MCI.hdf5':
        dataset_path = os.path.join(r"./datasets/mri/2_AD_MCI/")
    elif config.hdf5FileNametrain == 'train_MRIdata_2_AD_Normal.hdf5':
        dataset_path = os.path.join(r"./datasets/mri/2_AD_Normal/")
    else:
        dataset_path = os.path.join(r"./datasets/mri/2_MCI_Normal/")

    input_file_name = config.hdf5FileNametrain
    name_list = input_file_name.split("_")
    class_num=name_list[2]

    dataset_train_unlabelled, dataset_test, all_hdf5_data_train, all_hdf5_data_test, dataset_train_labelled = dataset.create_default_splits8020(
        dataset_path, config.hdf5FileNametrain,
        config.testhdf5FileName,
        config.idFileNametrain,
        config.testidFileName,config.num_less_label_data,class_num)

    data_provider = get_data_provider_by_path(config, dataset_train_unlabelled,dataset_train_labelled, dataset_test, all_hdf5_data_train,
                                              all_hdf5_data_test, 0)

    model = TripleGAN3D(config, data_provider, all_train_dir, 0, is_train=True)


    if config.train:
        total_start_time = time.time()
        print("Data provider train labelled images: ", data_provider.train_labelled.num_examples)
        print("Data provider train unlabelled images: ", data_provider.train_unlabelled.num_examples)
        model.train_all_epochs(config)
        total_training_time = time.time() - total_start_time
        # f.close()
        # fx.close()
        print("\n each fold Total training time for all epoches : %s  and %s seconds" % (str(timedelta(
            seconds=total_training_time)), total_training_time))

        fxx = open(all_train_dir[0] + '/timeReport_fold' + str(0) + '.txt', 'w')
        fxx.write("\n each fold Total training time for all epoches : %s  and %s seconds" % (str(timedelta(
            seconds=total_training_time)), total_training_time))
        fxx.write('\n')
        fxx.close()

    if config.test:
        if not config.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")

        model.test_and_record(all_result_file_name[0], 0,config,all_train_dir[0],data_provider.test, batch_size=config.batch_size)

    input_file_name = config.hdf5FileNametrain
    class_labels = []
    name_list = input_file_name.split("_")
    if int(name_list[2]) == 3:
        class_labels.append(name_list[3])
        class_labels.append(name_list[4])
        last_class = name_list[5].split(".")
        class_labels.append(last_class[0])
    else:
        class_labels.append(name_list[3])
        last_class = name_list[4].split(".")
        class_labels.append(last_class[0])

    if tf.gfile.Exists("./GANconfusionMatrixResults"):
        log.infov("./GANconfusionMatrixResults")
    else:
        os.makedirs("./GANconfusionMatrixResults")

    accuracy_10folds_all = []
    fold_write = 0
    for each_result_file_name in all_result_file_name:
        if fold_write < 1:
            accuracy, cr, cm = calculateConfusionMatrix(each_result_file_name, class_labels, all_train_dir[0])
        else:
            accuracy, cr, cm = calculateConfusionMatrix(each_result_file_name, class_labels, './train_dir')

        f = open("./GANconfusionMatrixResults/ConfusionMatrix-" + each_result_file_name + ".txt", 'w')
        log.info("Fold: {}".format(fold_write))
        f.write("Fold: {}\n".format(fold_write))
        f.write(
            '{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(config.hdf5FileNametrain, cr, cm))
        f.write("accuracy: {}\n".format(accuracy))
        log.info("accuracy: {}".format(accuracy))
        f.close()




if __name__ == '__main__':
  tf.app.run()
