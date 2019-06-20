import os
import time
import shutil
import platform
from datetime import timedelta

import numpy as np
import tensorflow as tf
from util import log
import tensorflow.contrib.slim as slim

import csv
TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))



class EvalManager(object):

  def __init__(self,train_dir):
    # collection of batches (not flattened)
    self._ids = []
    self._predictions = []
    self._groundtruths = []
    self.train_dir=train_dir



  # def add_batch(self, id, prediction, groundtruth):
  #
  #     # for now, store them all (as a list of minibatch chunks)
  #     self._ids.append(id)
  #     self._predictions.append(prediction)
  #     self._groundtruths.append(groundtruth)

  def compute_accuracy(self, pred, gt):
    correct_prediction = np.sum(np.argmax(pred[:, :-1], axis=1) == np.argmax(gt, axis=1))
    return float(correct_prediction) / pred.shape[0]

  def add_batch_new(self, prediction, groundtruth,n_classes):

    # for now, store them all (as a list of minibatch chunks)
    shold_be_batch_size = len(prediction)



    for index in range(shold_be_batch_size):
      # prediction_first_k_class = []
      # for i in range(n_classes):
      #   prediction_first_k_class.append(prediction[index][i])
      self._predictions.append(prediction[index])
      self._groundtruths.append(groundtruth[index])

  def add_batch(self, id, prediction, groundtruth):

    # for now, store them all (as a list of minibatch chunks)
    shold_be_batch_size = len(id)
    for index in range(shold_be_batch_size):
      self._ids.append(id[index])
      self._predictions.append(prediction[index])
      self._groundtruths.append(groundtruth[index])

  def report(self, result_file_name):
    # report L2 loss
    # log.info("Computing scores...")

    z = zip(self._predictions, self._groundtruths)
    u = list(z)

    if tf.gfile.Exists(self.train_dir + '/GANresults/'):
      log.infov("self.train_dir + '/GANresults/' exists")
    else:
      os.makedirs(self.train_dir + '/GANresults/')

    with open(self.train_dir + '/GANresults/' + result_file_name + '.csv', mode='w') as file:
      writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['actual_label', 'model_GAN'])
      for pred, gt in u:
        # gt_csv = np.argmax(gt)
        # pred_csv = np.argmax(pred)
        gt_csv = gt
        pred_csv = pred

        one_row = []

        one_row.append(str(gt_csv))
        one_row.append(str(pred_csv))

        writer.writerow(one_row)




class TripleGAN3D(object):
  def __init__(self, config,data_provider,all_train_dir,whichFoldData,is_train=True):
    """
    Class to implement networks base on this paper
    https://arxiv.org/pdf/1611.05552.pdf

    Args:
      data_provider: Class, that have all required data sets
      growth_rate: `int`, variable from paper
      depth: `int`, variable from paper
      total_blocks: `int`, paper value == 3
      keep_prob: `float`, keep probability for dropout. If keep_prob = 1
        dropout will be disables
      weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
      nesterov_momentum: `float`, momentum for Nesterov optimizer
      model_type: `str`, 'DenseNet3D' or 'DenseNet3D-BC'. Should model use
        bottle neck connections or not.
      dataset: `str`, dataset name
      should_save_logs: `bool`, should logs be saved or not
      should_save_model: `bool`, should model be saved or not
      renew_logs: `bool`, remove previous logs for current model
      reduction: `float`, reduction Theta at transition layer for
        DenseNets with bottleneck layers. See paragraph 'Compression'
        https://arxiv.org/pdf/1608.06993v3.pdf#4
      bc_mode: `bool`, should we use bottleneck layers and features
        reduction or not.
    """
    tf.reset_default_graph()
    self.data_provider          = data_provider
    self.data_shape             = data_provider.data_shape
    self.n_classes              = data_provider.n_classes
    self.depth                  = config.depth
    self.growth_rate            = config.growth_rate
    # how many features will be received after first convolution
    # value the same as in the original Torch code
    self.first_output_features  = config.growth_rate * 2
    self.total_blocks           = config.total_blocks
    self.layers_per_block       = (config.depth - (config.total_blocks + 1)) // config.total_blocks
    self.bc_mode = config.bc_mode
    # compression rate at the transition layers
    self.reduction              = config.reduction
    if not config.bc_mode:
      print("Build %s model with %d blocks, "
          "%d composite layers each." % (
            config.model_type, self.total_blocks, self.layers_per_block))
    if config.bc_mode:
      self.layers_per_block     = self.layers_per_block // 2
      print("Build %s model with %d blocks, "
          "%d bottleneck layers and %d composite layers each." % (
            config.model_type, self.total_blocks, self.layers_per_block,
            self.layers_per_block))
    print("Reduction at transition layers: %.1f" % self.reduction)

    self.keep_prob          = config.keep_prob
    self.weight_decay       = config.weight_decay
    self.nesterov_momentum  = config.nesterov_momentum
    self.model_type         = config.model_type
    self.dataset_name       = config.hdf5FileNametrain
    self.labeled_rate       = config.labeled_rate

    self.batches_step       = 0
    self.sequence_length    = 109
    self.crop_size          = (91,91)
    self.train_dir= all_train_dir[whichFoldData]
    self.whichFoldData=whichFoldData
    self.renew_logs=config.renew_logs
    self.n_z = config.n_z
    self.batch_size = config.batch_size

    self.weights = self.initialiseWeights()
    self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

    self.build_triple_GAN()



    self._initialize_session()

  ############ Preparing Mask ############

  # Preparing a binary label_mask to be multiplied with real labels
  def get_labeled_mask(self, labeled_rate, batch_size):
    labeled_mask = np.zeros([batch_size], dtype=np.float32)
    labeled_count = np.int(batch_size * labeled_rate)
    labeled_mask[range(labeled_count)] = 1.0
    np.random.shuffle(labeled_mask)
    return labeled_mask

  ############ Preparing Extended label ############

  def prepare_extended_label(self, label):
    # add extra label for fake data
    extended_label = tf.concat([tf.zeros([tf.shape(label)[0], 1]), label], axis=1)

    return extended_label

    ############ Defining losses ############

    # The total loss inculcates  D_L_Unsupervised + D_L_Supervised + G_feature_matching loss + G_R/F loss

  def build_loss(self, d_real, d_real_logits, d_fake, d_fake_logits, label, real_image, fake_image, d_real_feature_map,
                 d_fake_feature_map):

    extended_label = self.prepare_extended_label(label)

    ### Discriminator loss ###

    # Supervised loss -> which class the real data belongs to

    temp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_real_logits[:, 1:],
                                                      labels=extended_label[:, 1:])

    # Don't confuse labeled_rate with labeled_mask
    # Labeled_mask and temp are of same size = batch_size where temp is softmax
    # cross_entropy calculated over whole batch
    D_L_Supervised = tf.reduce_mean(temp)

    # D_L_Supervised = tf.reduce_sum(tf.multiply(temp, self.labeled_mask)) / tf.reduce_sum(self.labeled_mask)
    # Multiplying temp with labeled_mask gives supervised loss on labeled_mask
    # data only, calculating mean by dividing by no of labeled samples

    # Unsupervised loss -> R/F

    D_L_RealUnsupervised = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits[:, 0], labels=tf.zeros_like(d_real_logits[:, 0], dtype=tf.float32)))
    D_L_FakeUnsupervised = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits[:, 0], labels = tf.ones_like(d_fake_logits[:, 0], dtype=tf.float32)))


    all_var = tf.trainable_variables()
    d_var = [v for v in all_var if v.name.startswith('Discriminator')]

    l2_loss = tf.add_n(
      [tf.nn.l2_loss(var) for var in d_var])
    # total_d_loss = d_loss + l2_loss * self.weight_decay
    total_d_loss = 10 * D_L_Supervised + D_L_RealUnsupervised + D_L_FakeUnsupervised + l2_loss * self.weight_decay

    ### Generator loss ###

    # G_L_1 -> Fake data wanna be real

    G_L_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=d_fake_logits[:, 0], labels=tf.zeros_like(d_fake_logits[:, 0], dtype=tf.float32)))

    # G_L_2 -> Feature matching
    f_match = tf.constant(0., dtype=tf.float32)
    for i in range(len(d_real_feature_map)):
      f_match += tf.reduce_mean(tf.square(d_real_feature_map[i] - d_fake_feature_map[i]))

    G_L = G_L_1 + f_match

    GAN_loss = tf.reduce_mean(total_d_loss + G_L)

    # Classification accuracy
    self.all_preds = tf.argmax(d_real[:, 1:], 1)
    self.all_targets = tf.argmax(extended_label[:, 1:], 1)


    correct_prediction = tf.equal(tf.argmax(d_real[:, 1:], 1),
                                  tf.argmax(extended_label[:, 1:], 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return D_L_Supervised, D_L_RealUnsupervised, D_L_FakeUnsupervised, total_d_loss, G_L, GAN_loss, accuracy



  def build_triple_GAN(self):

    self._define_inputs_D_G()

    self.fake_image = self._build_graph_G(self.z_vector,phase_train=self.is_training)
    d_real_softmax, d_no_softmax_real,d_real_feature_map = self._build_graph_D(self.x_vector, reuse=False)
    d_fake_softmax, d_no_softmax_fake,d_fake_feature_map = self._build_graph_D(self.fake_image, reuse=True)

    # Compute the discriminator and generator loss
    # d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1-d_fake))
    # g_loss = -tf.reduce_mean(tf.log(d_fake))
    n=self.n_classes


    self.S_loss, self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss, self.GAN_loss, self.accuracy = \
      self.build_loss(d_real_softmax, d_no_softmax_real, d_fake_softmax, d_no_softmax_fake, self.labels, self.x_vector, self.fake_image,d_real_feature_map,d_fake_feature_map)

    # tf.summary.scalar("loss/accuracy", self.accuracy)
    # tf.summary.scalar("loss/GAN_loss", GAN_loss)
    # tf.summary.scalar("loss/S_loss", self.S_loss)
    # tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
    # tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
    # tf.summary.scalar("loss/d_loss_fake", tf.reduce_mean(d_loss_fake))
    # tf.summary.scalar("loss/g_loss", tf.reduce_mean(self.g_loss))
    # tf.summary.image("img/fake", fake_image)
    # tf.summary.image("img/real", self.image, max_outputs=1)
    # tf.summary.image("label/target_real", tf.reshape(self.label, [1, self.batch_size, n, 1]))
    log.warn('\033[93mSuccessfully loaded the model.\033[0m')

    # net_g_test = self._build_graph_G(self.z_vector, phase_train=False, reuse=True)

    para_g = [v for v in tf.trainable_variables() if v.name.startswith('Generator')]
    para_d = [v for v in tf.trainable_variables() if v.name.startswith('Discriminator')]

    # only update the weights for the discriminator network

    self.optimizer_op_d = tf.train.MomentumOptimizer(
      self.learning_rate, self.nesterov_momentum, use_nesterov=True).minimize(self.d_loss, var_list=para_d)
    # only update the weights for the generator network
    self.optimizer_op_g = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.5).minimize(self.g_loss, var_list=para_g)




  def initialiseWeights(self):

    weights = {}
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 3, 3, 512, 128], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 32, 64], initializer=xavier_init)
    weights['wg6'] = tf.get_variable("wg6", shape=[4, 4, 4, 1, 32], initializer=xavier_init)

    return weights

  def huber_loss(self, labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)





  def _initialize_session(self):
    """Initialize session, variables, saver"""

    config = tf.ConfigProto()
    # restrict model GPU memory utilization to min required
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    tf_ver = int(tf.__version__.split('.')[1])
    if TF_VERSION <= 0.10:
      self.sess.run(tf.initialize_all_variables())
      logswriter = tf.summary.SummaryWriter(self.train_dir)
    else:
      self.sess.run(tf.global_variables_initializer())
      logswriter = tf.summary.FileWriter
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
    self.summary_writer = logswriter(self.train_dir, self.sess.graph)
    self.summary_op = tf.summary.merge_all()



  # (Updated)
  def _count_trainable_params(self):
    total_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parametes = 1
      for dim in shape:
        variable_parametes *= dim.value
      total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

  @property
  def save_path(self):
    try:
      save_path = self._save_path
      model_path = self._model_path
    except AttributeError:
      save_path = self.train_dir
      if platform.python_version_tuple()[0] is '2':
        if not os.path.exists(save_path):
          os.makedirs(save_path)
      else:
        os.makedirs(save_path, exist_ok=True)
      model_path = os.path.join(save_path, 'model.chkpt')
      self._save_path = save_path
      self._model_path = model_path
    return save_path, model_path



  @property
  def model_identifier(self):
    return "{}_growth_rate={}_depth={}_seq_length={}_crop_size={}".format(
      self.model_type, self.growth_rate, self.depth, self.sequence_length,
      self.crop_size)

  # (Updated)
  def save_model(self, global_step=None):
    self.saver.save(self.sess, self.train_dir+'/', global_step=global_step)

  def load_model(self,which_check_point=None):
    """load the sess from the pretrain model

      Returns:
        start_epoch: the start step to train the model
    """
    # Restore the trianing model from the folder
    if which_check_point==None:
      ckpt = tf.train.get_checkpoint_state(self.save_path[0])
    else:
      ckpt = tf.train.get_checkpoint_state(self.save_path[0])
      ckpt.model_checkpoint_path = os.path.join(self.save_path[0],
                                                "-"+which_check_point)


    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      start_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      start_epoch = int(start_epoch) + 1
      print("Successfully load model from save path: %s and epoch: %s" 
          % (self.save_path[0], start_epoch))
      return start_epoch
    else:
      print("Training from scratch")
      return 0

  def log_one_metric(self, metric, epoch, prefix):

    summary = tf.Summary(value=[
      tf.Summary.Value(
        tag='loss_%s' % prefix, simple_value=float(metric))
    ])
    self.summary_writer.add_summary(summary, epoch)



  def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                        should_print=True):
    print(prefix + "mean cross_entropy: %f, mean accuracy: %f" % (
        loss, accuracy))
    summary = tf.Summary(value=[
      tf.Summary.Value(
        tag='loss_%s' % prefix, simple_value=float(loss)),
      tf.Summary.Value(
        tag='accuracy_%s' % prefix, simple_value=float(accuracy))
    ])
    self.summary_writer.add_summary(summary, epoch)

  # (Updated)
  def _define_inputs_D_G(self):
    shape = [None]
    shape.extend(self.data_shape)
    # self.videos = tf.placeholder(
    #   tf.float32,
    #   shape=shape,
    #   name='input_videos')
    self.labels = tf.placeholder(
      dtype=tf.float32,
      shape=[None, self.n_classes],
      name='labels')



    self.learning_rate = tf.placeholder(
      tf.float32,
      shape=[],
      name='learning_rate')

    self.recon_weight = tf.placeholder_with_default(
      tf.cast(1.0, tf.float32), [])


    self.x_vector = tf.placeholder(shape=shape, dtype=tf.float32)
    self.z_vector = tf.placeholder(shape=[None, self.n_z], dtype=tf.float32)
    # self.labeled_mask = tf.placeholder(dtype=tf.float32, name = 'labeled_mask', shape = [None])



    # (Updated)
  def composite_function(self, _input, out_features, kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
      # BN
      output = self.batch_norm(_input)
      # ReLU
      with tf.name_scope("ReLU"):
        output = tf.nn.relu(output)
      # convolution
      output = self.conv3d(
        output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
      output = self.dropout(output)
    return output

  # (Updated)
  def bottleneck(self, _input, out_features):
    with tf.variable_scope("bottleneck"):
      output = self.batch_norm(_input)
      with tf.name_scope("ReLU"):
        output = tf.nn.relu(output)
      inter_features = out_features * 4
      output = self.conv3d(
        output, out_features=inter_features, kernel_size=1,
        padding='VALID')
      output = self.dropout(output)
    return output

  # (Updated)
  def add_internal_layer(self, _input, growth_rate):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not self.bc_mode:
      comp_out = self.composite_function(
        _input, out_features=growth_rate, kernel_size=3)
    elif self.bc_mode:
      bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
      comp_out = self.composite_function(
        bottleneck_out, out_features=growth_rate, kernel_size=3)
    # concatenate _input with out from composite function
    with tf.name_scope("concat"):
      if TF_VERSION >= 1.0:
          output = tf.concat(axis=4, values=(_input, comp_out))
      else:
        output = tf.concat(4, (_input, comp_out))
    return output

  # (Updated)
  def add_block(self, _input, growth_rate, layers_per_block):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
      with tf.variable_scope("layer_%d" % layer):
        output = self.add_internal_layer(output, growth_rate)
    return output

  # (Updated)
  def transition_layer(self, _input, pool_depth=2):
    """Call H_l composite function with 1x1 kernel and pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * self.reduction)
    output = self.composite_function(
      _input, out_features=out_features, kernel_size=1)
    # run pooling
    with tf.name_scope("pooling"):
      output = self.pool(output, k=2, d=pool_depth)
    return output

  # (Updated)
  def trainsition_layer_to_classes(self, _input,out_feature_map):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide pooling
    - FC layer multiplication
    """
    # BN
    output = self.batch_norm(_input)
    # ReLU
    with tf.name_scope("ReLU"):
      output = tf.nn.relu(output)
    # pooling
    last_pool_kernel_width = int(output.get_shape()[-2])
    last_pool_kernel_height = int(output.get_shape()[-3])
    last_sequence_length = int(output.get_shape()[1])
    with tf.name_scope("pooling"):
      output = self.pool(output, k = last_pool_kernel_height,
                         d = last_sequence_length,
                         width_k = last_pool_kernel_width,
                         k_stride_width = last_pool_kernel_width)
    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    out_feature_map.append(output)
    W = self.weight_variable_xavier(
      [features_total, self.n_classes+1], name='W')
    bias = self.bias_variable([self.n_classes+1])
    logits = tf.matmul(output, W) + bias

    return logits,out_feature_map
  
  # (Updated)
  def conv3d(self, _input, out_features, kernel_size,
         strides=[1, 1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = self.weight_variable_msra(
      [kernel_size, kernel_size, kernel_size, in_features, out_features],
      name='kernel')
    with tf.name_scope("3DConv"):
      output = tf.nn.conv3d(_input, kernel, strides, padding)
    return output

  # (Updated)
  def pool(self, _input, k, d=2, width_k=None, type='avg', k_stride=None, d_stride=None, k_stride_width=None):
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

  # (Updated)
  def batch_norm(self, _input):
    with tf.name_scope("batch_normalization"):
      output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=self.is_training,
        updates_collections=None)
    return output

  # (Updated)
  def dropout(self, _input):
    if self.keep_prob < 1:
      with tf.name_scope('dropout'):
        output = tf.cond(
          self.is_training,
          lambda: tf.nn.dropout(_input, self.keep_prob),
          lambda: _input
        )
    else:
      output = _input
    return output

  # (Updated)
  def weight_variable_msra(self, shape, name):
    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=tf.contrib.layers.variance_scaling_initializer())

  # (Updated)
  def weight_variable_xavier(self, shape, name):
    return tf.get_variable(
      name,
      shape=shape,
      initializer=tf.contrib.layers.xavier_initializer())


  def bias_variable(self, shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


  def _build_graph_G(self, z, phase_train=True):
    strides = [1, 2, 2, 2, 1]
    weights=self.weights
    batch_size=self.batch_size


    with tf.variable_scope("Generator"):
      z = tf.reshape(z, (self.batch_size, 1, 1, 1, self.n_z))
      g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size, 4, 3, 3, 512), strides=[1, 1, 1, 1, 1],
                                   padding="VALID")
      # g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size, 4, 3, 3, 512), strides=strides,
      #                              padding="SAME")
      g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
      g_1 = tf.nn.relu(g_1)

      g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size, 7, 6, 6, 256), strides=strides, padding="SAME")
      g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
      g_2 = tf.nn.relu(g_2)

      g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size, 14, 12, 12, 128), strides=strides, padding="SAME")
      g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
      g_3 = tf.nn.relu(g_3)

      g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size, 28, 23, 23, 64), strides=strides, padding="SAME")
      g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
      g_4 = tf.nn.relu(g_4)

      g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size, 55, 46, 46, 32), strides=strides, padding="SAME")
      g_5 = tf.contrib.layers.batch_norm(g_5, is_training=phase_train)
      g_5 = tf.nn.relu(g_5)

      g_6 = tf.nn.conv3d_transpose(g_5, weights['wg6'], (batch_size, 109, 91, 91, 1), strides=strides, padding="SAME")

      g_6 = tf.nn.tanh(g_6)

    print g_1, 'g1'
    print g_2, 'g2'
    print g_3, 'g3'
    print g_4, 'g4'
    print g_5, 'g5'
    print g_6, 'g6'

    return g_6





  def _build_graph_D(self,mri,reuse=False):
    growth_rate = self.growth_rate
    layers_per_block = self.layers_per_block
    # first - initial 3 x 3 x 3 conv to first_output_features


    with tf.variable_scope("Discriminator", reuse=reuse):
      out_feature_map=[]
      with tf.variable_scope("Initial_convolution"):
        output = self.conv3d(
          mri,
          out_features=self.first_output_features,
          kernel_size=7,
          strides=[1, 1, 2, 2, 1])

        # first pooling
        output = self.pool(output, k=3, d=3, k_stride=2, d_stride=1)


      # add N required blocks
      for block in range(self.total_blocks):
        with tf.variable_scope("Block_%d" % block):
          output = self.add_block(output, growth_rate, layers_per_block)
          out_feature_map.append(output)

        # last block exist without transition layer
        if block != self.total_blocks - 1:
          with tf.variable_scope("Transition_after_block_%d" % block):
            # pool_depth = 1 if block == 0 else 2
            pool_depth = 2
            output = self.transition_layer(output, pool_depth)
            out_feature_map.append(output)


      with tf.variable_scope("Transition_to_classes"):
        logits,out_feature_map = self.trainsition_layer_to_classes(output,out_feature_map)

    logits_softmax = tf.nn.softmax(logits)
    return logits_softmax, logits,out_feature_map


  # (Updated)
  def train_all_epochs(self, config):
    n_epochs           = config.max_training_steps
    init_learning_rate = config.learning_rate_d
    batch_size         = config.batch_size
    reduce_lr_epoch_1  = config.reduce_lr_epoch_1
    reduce_lr_epoch_2  = config.reduce_lr_epoch_2


    # Restore the model if we have
    start_epoch = self.load_model()
    # to illustrate overfitting with accuracy and loss later
    # f = open(self.train_dir +'/accuracy.txt', 'w')
    # f.write('epoch, train_acc, test_acc\n')
    #
    # fx = open(self.train_dir +'/loss.txt', 'w')
    # fx.write('epoch, train_loss, test_loss\n')

    
    # Start training 
    for epoch in range(start_epoch, n_epochs + 1):
      print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
      start_time = time.time()
      learning_rate = init_learning_rate
      # Update the learning rate according to the decay parameter
      if epoch >= reduce_lr_epoch_1 and epoch < reduce_lr_epoch_2:
        learning_rate = learning_rate / 10
        print("Decrease learning rate, new lr = %f" % learning_rate)
      elif epoch >= reduce_lr_epoch_2:
        learning_rate = learning_rate / 100
        print("Decrease learning rate, new lr = %f" % learning_rate)

      recon_weight= min(max(0, (1500 - epoch) / 1500), 1.0) * 10
      # recon_weight = ((1500 - epoch) / 1500) * 10

      print("Training...")



      mean_s_loss, mean_accuracy,mean_G_loss, mean_D_loss,mean_D_real_loss,mean_D_fake_loss,mean_gan_loss,lr,recon_w = self.train_one_epoch(
        self.data_provider.train_labelled, batch_size, learning_rate,recon_weight,epoch)

      self.log_one_metric(mean_s_loss, epoch, prefix='train_S_loss')
      self.log_one_metric(mean_G_loss, epoch, prefix='train_G_loss')
      self.log_one_metric(mean_D_real_loss, epoch, prefix='train_D_real_loss')
      self.log_one_metric(mean_D_fake_loss, epoch, prefix='train_D_fake_loss')
      self.log_one_metric(mean_gan_loss, epoch, prefix='train_gan_loss')

      self.log_loss_accuracy(mean_D_loss, mean_accuracy, epoch, prefix='train_D')


      summary = tf.Summary(value=[
        tf.Summary.Value(
          tag='learning_rate', simple_value=float(lr))
      ])

      self.summary_writer.add_summary(summary, epoch)

      summary = tf.Summary(value=[
        tf.Summary.Value(
          tag='recon_weight', simple_value=float(recon_w))
      ])

      self.summary_writer.add_summary(summary, epoch)

      print("Validation...")
      mean_d_loss_real,mean_accuracy,mean_s_loss = self.test(
        self.data_provider.test, batch_size)

      self.log_one_metric(mean_s_loss, epoch, prefix='test_S_loss')
      self.log_one_metric(mean_d_loss_real, epoch, prefix='test_D_real_loss')
      self.log_loss_accuracy(mean_d_loss_real, mean_accuracy, epoch, prefix='test_D')

      time_per_epoch = time.time() - start_time
      seconds_left = int((n_epochs - epoch) * time_per_epoch)
      print("Time per epoch: %s, Est. complete in: %s" % (
        str(timedelta(seconds=time_per_epoch)),
        str(timedelta(seconds=seconds_left))))

      self.save_model(global_step=epoch)
      self.summary_writer.add_summary(summary, global_step=epoch)
      self.summary_op = tf.summary.merge_all()




  # (Updated)
  def train_one_epoch(self, data, batch_size, learning_rate,recon_weight,epoch):
    num_examples = data.num_examples
    total_gan_loss = []
    total_D_fake_loss = []
    total_D_real_loss = []
    total_D_loss = []
    total_G_loss = []
    total_s_loss = []
    total_accuracy = []


    z = np.random.normal(0, 1, size=[batch_size, self.n_z]).astype(np.float32)

    for i in range(num_examples // batch_size):
      # videos size is (numpy array):
      #   [batch_size, sequence_length, width, height, channels]
      # labels size is (numpy array):
      #   [batch_size, num_classes]


      mris, labels = data.next_batch(batch_size)

      # mask = self.get_labeled_mask(self.labeled_rate, batch_size)



      fetch = [self.GAN_loss,self.d_loss_fake,self.d_loss_real,self.accuracy,
                 self.d_loss, self.g_loss, self.S_loss,
                 self.all_preds, self.all_targets,
                 self.fake_image, self.learning_rate,self.recon_weight]



      gan_loss_per_batch,d_loss_fake_per_batch,d_loss_real_per_batch, accuracy_per_batch,d_loss_per_batch,g_loss_per_batch,s_loss_per_batch,predicts_per_batch,gt_per_batch,fake_image_batch,lr,recon_w = self.sess.run(fetch, feed_dict={self.z_vector: z, self.x_vector: mris, self.learning_rate: learning_rate, self.recon_weight: recon_weight,self.is_training: True,self.labels:labels})

      if epoch % (2) > 0:
          # Train the generator
          self.sess.run([self.optimizer_op_g], feed_dict={self.z_vector: z, self.x_vector: mris, self.learning_rate: learning_rate, self.recon_weight: recon_weight,self.is_training: True,self.labels:labels})
          print("epoch:"+str(epoch)+"update generater parameters, optimize G")
      else:
          # Train the discriminator
          self.sess.run([self.optimizer_op_d], feed_dict={self.z_vector: z, self.x_vector: mris, self.learning_rate: learning_rate, self.recon_weight: recon_weight,self.is_training: True,self.labels:labels})
          print("epoch:" + str(epoch) + "update discriminator parameters, optimize D")

      total_gan_loss.append(gan_loss_per_batch)
      total_D_fake_loss.append(d_loss_fake_per_batch)
      total_D_real_loss.append(d_loss_real_per_batch)
      total_D_loss.append(d_loss_per_batch)
      total_G_loss.append(g_loss_per_batch)
      total_s_loss.append(s_loss_per_batch)
      total_accuracy.append(accuracy_per_batch)



      self.batches_step += 1
      # self.log_loss_accuracy(
      #     loss, accuracy, self.batches_step, prefix='per_batch',
      #     should_print=False)




    mean_gan_loss = np.mean(total_gan_loss)
    mean_D_fake_loss = np.mean(total_D_fake_loss)
    mean_D_real_loss = np.mean(total_D_real_loss)

    mean_D_loss = np.mean(total_D_loss)
    mean_G_loss = np.mean(total_G_loss)
    mean_accuracy = np.mean(total_accuracy)

    mean_s_loss = np.mean(total_s_loss)


    return mean_s_loss, mean_accuracy,mean_G_loss, mean_D_loss,mean_D_real_loss,mean_D_fake_loss,mean_gan_loss,lr,recon_w

  # (Updated)
  def test(self, data, batch_size):
    num_examples = data.num_examples

    total_d_loss_real= []
    total_accuracy= []

    total_s_loss= []

    for i in range(num_examples // batch_size):
      batch = data.next_batch(batch_size)
      feed_dict = {
        self.x_vector: batch[0],
        self.labels: batch[1],
        # self.z_vector: z,
        self.is_training: False,
      }


      fetches = [self.d_loss_real, self.accuracy,self.S_loss,self.all_preds,self.all_targets]
      d_loss_real_value, accuracy_value, S_loss_value,predicts,ground_truth = self.sess.run(fetches, feed_dict=feed_dict)


      total_d_loss_real.append(d_loss_real_value)
      total_accuracy.append(accuracy_value)

      total_s_loss.append(S_loss_value)

    mean_d_loss_real = np.mean(total_d_loss_real)
    mean_accuracy = np.mean(total_accuracy)


    mean_s_loss = np.mean(total_s_loss)


    return mean_d_loss_real,mean_accuracy,mean_s_loss

  def test_and_record(self, result_file_name, whichFoldData,config,train_dir, data,batch_size):
    evaler = EvalManager(self.train_dir)
    num_examples = data.num_examples
    total_loss = []
    total_accuracy = []
    batch_size=1
    # z = np.random.normal(0, 1, size=[batch_size, self.n_z]).astype(np.float32)

    for i in range(num_examples // batch_size):
      batch = data.next_batch(batch_size)
      feed_dict = {
        self.x_vector: batch[0],
        self.labels: batch[1],
        # self.z_vector: z,
        self.is_training: False,
      }



      fetches = [self.accuracy,self.all_preds,self.all_targets]
      accuracy,prediction,ground_truth = self.sess.run(fetches, feed_dict=feed_dict)

      # total_loss.append(s_loss)
      total_accuracy.append(accuracy)
      evaler.add_batch_new(prediction, ground_truth,self.n_classes)

    # mean_loss = np.mean(total_loss)
    mean_accuracy = np.mean(total_accuracy)
    evaler.report(result_file_name)

    log.info("final test accuracy: {}".format(mean_accuracy))

    # return mean_loss, mean_accuracy

