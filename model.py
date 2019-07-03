from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import huber_loss
from util import log
from generator import Generator
from discriminator import Discriminator
import numpy as np
import re
# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

class Model(object):

    def __init__(self, config, growth_rate, depth,
                 total_blocks, keep_prob,
                  nesterov_momentum, model_type,scope,
                 debug_information=False,
                 is_train=True,
                 reduction=1.0,
                 bc_mode=False):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.h = self.config.h
        self.w = self.config.w
        self.c = self.config.c
        self.num_class = self.config.num_class
        self.n_z = config.n_z
        self.scope=scope


        # create placeholders for the input


        # tf.summary.scalar("loss/recon_wieght", self.recon_weight)

        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob

        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type




    def get_feed_dict(self, batch_chunk, is_training=None):
        fd = {
            self.images: batch_chunk['image'],  # [bs, h, w, c]
            self.labels: batch_chunk['label'],  # [bs, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        # Weight annealing
        # if step is not None:
        #     fd[self.recon_weight] = min(max(0, (1500 - step) / 1500), 1.0)*10
        return fd






    def build(self, images,labels, weight_decay,is_train=True):

        n = self.num_class

        # build loss and accuracy {{{
        def build_loss(prediction, logits):
            # alpha = 0.9

            # Discriminator/classifier loss
            """Add L2Loss to all the trainable variables.

             Add summary for "Loss" and "Loss/avg".
             Args:
               logits: Logits from inference().
               labels: Labels from distorted_inputs or inputs(). 1-D tensor
                       of shape [batch_size]

             Returns:
               Loss tensor of type float.
             """
            # Calculate the average cross entropy loss across the batch.

            cross_entropy_mean_each_batch = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy_per_example'), name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean_each_batch)
            tf.add_to_collection('prediction_label', prediction)
            tf.add_to_collection('target_label', labels)

            correct_prediction_each_batch = tf.equal(
                tf.argmax(prediction, 1),
                tf.argmax(labels, 1))
            accuracy_each_batch = tf.reduce_mean(tf.cast(correct_prediction_each_batch, tf.float32))
            return cross_entropy_mean_each_batch, correct_prediction_each_batch, accuracy_each_batch, tf.add_n(tf.get_collection('losses'), name='total_loss')
        # }}}




        # Discriminator {{{
        # =========
        D = Discriminator('Discriminator', self.num_class,self.h, self.w, self.c,self.growth_rate, self.depth,
                 self.total_blocks, self.keep_prob,
                  self.model_type,
                 is_train=True,
                 reduction=self.reduction,
                 bc_mode=self.bc_mode)
        prediction, logits = D(images)
        self.all_preds = prediction
        self.all_targets = labels

        # }}}

        self.cross_entropy_mean_each_batch, self.correct_prediction_each_batch, self.accuracy_each_batch, total_loss_all_batches = \
            build_loss(prediction, logits)

        tf.summary.scalar("accuracy_train_each_batch", self.accuracy_each_batch)
        # tf.summary.scalar("loss/correct_prediction",  self.correct_prediction)
        tf.summary.scalar("loss/cross_entropy_each_batch", self.cross_entropy_mean_each_batch)
        # tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
        # tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
        # tf.summary.image("img/fake", fake_image)
        # tf.summary.image("img/real", self.image, max_outputs=1)
        # tf.summary.image("label/target_real", tf.reshape(self.label, [1, self.batch_size, n, 1]))
        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', self.scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')

        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        loss_l2_regular = total_loss + l2_loss * weight_decay

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)



        prediction_all = tf.get_collection('prediction_label', self.scope)
        labels_all = tf.get_collection('target_label', self.scope)

        correct_prediction_all_batch = tf.equal(
            tf.argmax(prediction_all, 1),
            tf.argmax(labels_all, 1))
        accuracy_all_batch = tf.reduce_mean(tf.cast(correct_prediction_all_batch, tf.float32))

        return loss_l2_regular,self.accuracy_each_batch, accuracy_all_batch,prediction_all,labels_all

