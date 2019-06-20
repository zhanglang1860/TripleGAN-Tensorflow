import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
from util import log
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import re
TOWER_NAME = 'tower'


TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


def print_info(name, shape, activation_fn):
    log.info('{}{} {}'.format(
        name,  '' if activation_fn is None else ' ('+activation_fn.__name__+')',
        shape))


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def create_variable(name, shape, initializer,
    dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
            initializer=initializer, trainable=trainable)
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def instance_norm(input):
    """
    Instance normalization
    """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def norm_and_act(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    """
    Apply normalization and/or activation function
    """
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True,
                    updates_collections=None,
                )
            elif norm == 'instance':
                _ = instance_norm(_, is_train)
            elif norm == 'None':
                _ = _
            else:
                raise NotImplementedError
    return _


def conv2d(input, output_shape, is_train, info=False, k=4, s=2, stddev=0.01,
           activation_fn=lrelu, norm='batch', name="conv2d"):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k, k], stride=s, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def batch_norm(_input,is_training):
    with tf.name_scope("batch_normalization"):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=is_training,
            updates_collections=None)

    return output

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))



def composite_function(_input, out_features, is_training,keep_prob,kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = batch_norm(_input,is_training=False)
        # ReLU
        with tf.name_scope("ReLU"):
            output = tf.nn.relu(output)

        # convolution
        output = conv3d_denseNet(
            output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        output = dropout(output,keep_prob,is_training)
        _activation_summary(output)
    return output

def dropout(_input,keep_prob,is_training):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
        )
    else:
        output = _input
    return output



def bottleneck(is_training,_input, out_features,keep_prob):
    with tf.variable_scope("bottleneck"):
        output = batch_norm(_input, is_training)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = conv3d_denseNet(
            output, out_features=inter_features, kernel_size=1,
            padding='VALID')
        output = dropout(output,keep_prob,is_training)
        _activation_summary(output)
    return output

def avg_pool(_input, k):
    ksize = [1, k, k, k,1]
    strides = [1, k, k,k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool3d(_input, ksize, strides, padding)
    return output


def avg_pool_final(_input, a,b,c):
    ksize = [1, a,b,c,1]
    strides = [1, a,b,c, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool3d(_input, ksize, strides, padding)
    return output

def pool(_input, k, d=2, width_k=None, type='avg', k_stride=None, d_stride=None, k_stride_width=None):
    if not width_k: width_k = k
    ksize = [1, d, k, width_k, 1]
    if not k_stride: k_stride = k
    if not k_stride_width: k_stride_width = k_stride
    if not d_stride: d_stride = d
    strides = [1, d_stride, k_stride, k_stride_width, 1]
    padding = 'SAME'
    if type is 'max':
      output = tf.nn.max_pool3d(_input, ksize, strides, padding)
    elif type is 'avg':
      output = tf.nn.avg_pool3d(_input, ksize, strides, padding)
    else:
      output = None
    return output

def transition_layer(_input,is_training,keep_prob,reduction, pool_depth=2):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(
        _input, out_features=out_features, is_training=is_training, keep_prob=keep_prob, kernel_size=1)
    # run average pooling
    # if min(int(output.get_shape()[1]),int(output.get_shape()[2]),int(output.get_shape()[3]))>1:
    with tf.name_scope("pooling"):
        output = pool(output, k=2, d=pool_depth)

    _activation_summary(output)
    return output

def bias_variable(shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


def weight_variable_xavier(shape, name):
    return tf.get_variable(
        name,
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())


def transition_layer_to_classes(_input, n_classes,_is_train):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    # BN
    output = batch_norm(_input,_is_train)
    # ReLU
    output = tf.nn.relu(output)
    last_pool_kernel_width = int(output.get_shape()[-2])
    last_pool_kernel_height = int(output.get_shape()[-3])
    last_sequence_length = int(output.get_shape()[1])
    with tf.name_scope("pooling"):
        output = pool(output, k=last_pool_kernel_height,
                           d=last_sequence_length,
                           width_k=last_pool_kernel_width,
                           k_stride_width=last_pool_kernel_width)

    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    W = _variable_with_weight_decay('weights', [features_total, n_classes],
                                          stddev=0.01, wd=None)
    bias = _variable_on_cpu('biases', [n_classes],
                              tf.constant_initializer(0.0))
    logits = tf.matmul(output, W) + bias
    _activation_summary(logits)
    return logits


def add_internal_layer(keep_prob, is_training,_input, growth_rate,bc_mode):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not bc_mode:
        comp_out = composite_function(
            _input, out_features=growth_rate, is_training=is_training, keep_prob=keep_prob, kernel_size=3)
    elif bc_mode:
        bottleneck_out = bottleneck(is_training,_input, out_features=growth_rate,keep_prob=keep_prob)
        comp_out = composite_function(
            bottleneck_out, out_features=growth_rate, is_training=is_training, keep_prob=keep_prob, kernel_size=3)
    # concatenate _input with out from composite function
    if TF_VERSION >= 1.0:
        output = tf.concat(axis=4, values=(_input, comp_out))
    else:
        output = tf.concat(4, (_input, comp_out))
    return output

def add_block(keep_prob, is_training,_input, growth_rate, layers_per_block,bc_mode):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
        with tf.variable_scope("layer_%d" % layer):
            output = add_internal_layer(keep_prob,is_training, output, growth_rate,bc_mode)
    return output



def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer())

def conv3d_denseNet(_input, out_features, kernel_size,
                    strides=[1, 1, 1, 1,1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = _variable_with_weight_decay('3dConvKernelWeights',
                                         shape=[kernel_size, kernel_size,kernel_size, in_features, out_features],
                                         stddev=0.01,
                                         wd=None)

    output = tf.nn.conv3d(_input, filter = kernel, strides = strides, padding = padding)
    return output



def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    # dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.variance_scaling_initializer())
  return var




def conv3d_denseNet_first_layer(_input, out_features, kernel_size,
                    strides=[1, 2, 2,2,1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = _variable_with_weight_decay('kernel_weights',
                                         shape=[kernel_size, kernel_size,kernel_size, in_features, out_features],
                                         stddev=0.01,
                                         wd=None)
    output = tf.nn.conv3d(_input, filter = kernel, strides = strides, padding = padding)
    _activation_summary(output)
    return output

def conv2d_denseNet(_input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

def deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
             activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def bilinear_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
                      activation_fn=lrelu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _



def depthwise_conv2d(input, channel_multiplier, is_train, info=False, k=3, s=1,
                      activation_fn=lrelu, norm='batch', name='depthwiseConv2d'):
    inputs_shape = input.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(name):
        filter = create_variable("filter", shape=[k, k,
                                                  in_channels, channel_multiplier],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))


        _ = tf.nn.depthwise_conv2d(input,filter, strides=[1, s, s, 1], padding='SAME',rate=[1, 1])
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def grouped_conv2d_Discriminator(input, num_outputs, groups, is_train, info=False, k=3, s=1,
                      activation_fn=lrelu, norm='batch', name='groupedConv2d'):
    with tf.variable_scope(name):
        _ = grouped_convolution(input,num_outputs, [k, k],groups, stride=s)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def grouped_conv2d_Discriminator_valid(input, num_outputs, groups, is_train, info=False, k=3, s=2,
                      activation_fn=lrelu, norm='batch', name='groupedConv2d'):
    with tf.variable_scope(name):
        _ = grouped_convolution(input,num_outputs, [k, k],groups, stride=s,padding='VALID')
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def grouped_conv2d_Discriminator_one(input, num_outputs, groups, is_train, info=False, a=1,b=2, s=1,
                      activation_fn=lrelu, norm='batch', name='groupedConv2dKdifferent'):
    with tf.variable_scope(name):
        _ = grouped_convolution(input,num_outputs, [a, b],groups, stride=s,padding='VALID')
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def grouped_conv2d(input, num_outputs, groups, is_train, info=False, k=3, s=1,
                      activation_fn=lrelu, norm='batch', name='groupedConv2d'):
    with tf.variable_scope(name):
        _ = grouped_convolution(input,num_outputs, [k, k],groups, stride=s,padding='SAME')
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def grouped_conv2d_GsoP(input, num_outputs, groups, is_train=True, info=False, a=1, b=1,s=1,
                       name='groupedConv2d'):
    with tf.variable_scope(name):
        _ = grouped_convolution(input,num_outputs, [a, b],groups, stride=s,padding='VALID')
        # _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(),activation_fn=None)
    return _

def batch_norm_GAN(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [1])
    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=epsilon)


def _covariance(x, diag):
    """Defines the covariance operation of a matrix.

    Args:
    x: a matrix Tensor. Dimension 0 should contain the number of examples.
    diag: if True, it computes the diagonal covariance.

    Returns:
    A Tensor representing the covariance of x. In the case of
    diagonal matrix just the diagonal is returned.
    """

    f = tf.transpose(x, [2, 0, 1])
    fshape = f.get_shape().as_list()
    g = tf.reshape(f, [-1])
    shape_size = fshape[1] * fshape[2]
    covariance_matrix_shape = np.dtype('int32').type(shape_size)
    h = tf.reshape(g, [-1, covariance_matrix_shape])
    h = tf.transpose(h, [1, 0])

    num_points = math_ops.to_float(array_ops.shape(h)[0])
    h -= math_ops.reduce_mean(h, 0, keepdims=True)
    if diag:
        cov = math_ops.reduce_sum(math_ops.square(h), 0, keepdims=True) / (num_points - 1)
    else:
        cov = math_ops.matmul(h, h, transpose_a=True) / (num_points - 1)

    batch_norm(cov)
    return cov



def Global_Covariance_Matrix(x, diag):
    x = norm_and_act(x, is_train=True, norm='batch', activation_fn=lrelu)
    covariance_matrix_shape = x.get_shape().as_list()
    for i in range(0, covariance_matrix_shape[0]):
        each_image_covariance_matrix = _covariance(x[i], diag)
        each_image_covariance_matrix = tf.reshape(each_image_covariance_matrix,
                                                  [1, covariance_matrix_shape[3], covariance_matrix_shape[3]])
        if i == 0:
            result = each_image_covariance_matrix
        else:
            result = tf.concat([result, each_image_covariance_matrix], axis=0)
    return result



def excitation_layer(input_x,out_dim,orignialInput,is_train=True,name="GsoPexcitation"):
    with tf.variable_scope(name):
        excitation = norm_and_act(input_x, is_train, norm='batch', activation_fn=lrelu)
        excitation = slim.conv2d(excitation, out_dim, [1, 1], stride=1, activation_fn=None)
        excitation = tf.sigmoid(excitation)
        # excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = orignialInput * excitation
    return scale


def squeeze_excitation_layer(input_x, is_train=True, name="GsoP"):
    with tf.variable_scope(name):
        orignialInput = input_x
        number_filters_each_group=8
        out_dim = input_x.get_shape().as_list()[3]
        covariance_matrix_shape = out_dim / number_filters_each_group
        covariance_matrix_shape = np.dtype('int32').type(covariance_matrix_shape)
        squeeze = slim.conv2d(input_x, covariance_matrix_shape, [1, 1], stride=1, activation_fn=None)

        squeeze = Global_Covariance_Matrix(squeeze, False)

        squeeze = tf.reshape(squeeze, [-1, 1, covariance_matrix_shape, covariance_matrix_shape])


        excitation = grouped_conv2d_GsoP(squeeze, int(number_filters_each_group/2) * covariance_matrix_shape, covariance_matrix_shape, is_train=True, info=True,a = 1,  b=covariance_matrix_shape,
                           s=1)
        # excitation = grouped_conv_2d(squeeze, 4, [1, covariance_matrix_shape],  strides=1, padding='VALID', name=name)

        scale = excitation_layer(excitation, out_dim, orignialInput, name=name)
    return scale



def nn_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
                activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_nearest_neighbor(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def fc(input, output_shape, is_train, info=False, norm='batch',
       activation_fn=lrelu, name="fc"):
    with tf.variable_scope(name):
        _ = slim.fully_connected(input, output_shape, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _
