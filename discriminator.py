#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from ops import conv3d_denseNet
from ops import add_block
from util import log
from ops import depthwise_conv2d
from ops import fc
from ops import transition_layer
from ops import transition_layer_to_classes
from ops import grouped_conv2d_Discriminator_one
from ops import conv3d_denseNet_first_layer


class Discriminator(object):
    def __init__(self,
                 name,
                 num_class,
                 h,
                 w,
                 c,
                 growth_rate,
                 depth,
        total_blocks,
                 keep_prob,
         model_type,
                 is_train=True,
                 reduction=1.0,
                 bc_mode=False):
        self.name = name
        self._num_class = num_class
        self._is_train = is_train
        self._reuse = False
        self._h = h
        self._w = w
        self._c = c
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
        self.model_type = model_type






    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse) as scope:
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')

            growth_rate = self.growth_rate
            layers_per_block = self.layers_per_block
            # first - initial 3 x 3 conv to first_output_features
            with tf.variable_scope("Initial_convolution"):
                output = conv3d_denseNet_first_layer(
                    input,
                    out_features=self.first_output_features,
                    kernel_size=3)

            # add N required blocks
            for block in range(self.total_blocks):
                with tf.variable_scope("Block_%d" % block):
                    output = add_block(self.keep_prob,self._is_train, output, growth_rate, layers_per_block, self.bc_mode)
                # last block exist without transition layer
                if block != self.total_blocks - 1:
                    with tf.variable_scope("Transition_after_block_%d" % block):
                        output = transition_layer(output,self._is_train,self.keep_prob, self.reduction)

            with tf.variable_scope("Transition_to_classes"):
                logits = transition_layer_to_classes(output,self._num_class,self._is_train)
            prediction = tf.nn.softmax(logits)
            if not self._reuse: 
                log.info('discriminator output {}'.format(logits.shape.as_list()))
            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return prediction, logits
