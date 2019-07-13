import tensorflow as tf
import numpy as np
import math
from .aux import get_var_wrap

def tt_conv_full_3d(inp,
                 window,
                 inp_ch_modes,              
                 out_ch_modes,
                 ranks,
                 strides=[1, 1],
                 padding='SAME',
                 filters_initializer=tf.contrib.layers.variance_scaling_initializer(),
                 filters_regularizer=None,
                 cores_initializer=tf.contrib.layers.variance_scaling_initializer(),
                 cores_regularizer=None,
                 biases_initializer=tf.zeros_initializer,
                 biases_regularizer=None,
                 trainable=True,
                 cpu_variables=False,        
                 scope=None):
    """ tt-conv-layer (convolution of full input tensor with tt-filters (make tt full then use conv2d))
    Args:
        inp: input tensor, float - [batch_size, H, W, C]
        window: convolution window size, list [wH, wW]
        inp_ch_modes: input channels modes, np.array (int32) of size d
        out_ch_modes: output channels modes, np.array (int32) of size d
        ranks: tt-filters ranks, np.array (int32) of size (d + 1)        
        strides: strides, list of 2 ints - [sx, sy] 
        padding: 'SAME' or 'VALID', string
        filters_initializer: filters init function
        filters_regularizer: filters regularizer function
        cores_initializer: cores init function, could be a list of functions for specifying different function for each core
        cores_regularizer: cores regularizer function, could be a list of functions for specifying different function for each core
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """                

    with tf.variable_scope(scope):
        # inp_shape = inp.get_shape().as_list()
        # inp_h, inp_w, inp_ch = inp_shape[4]
        # # tmp = tf.reshape(inp, [-1, inp_h, inp_w, inp_ch])
        
        filters_shape = [window[0], window[1], window[2],1, ranks[0]]

        filters = get_var_wrap('filters',
                                   shape=filters_shape,
                                   initializer=filters_initializer,
                                   regularizer=filters_regularizer,
                                   trainable=trainable,
                                   cpu_variable=cpu_variables)
        d = inp_ch_modes.size
        
        cores = []
        for i in range(d):
            
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            if type(cores_regularizer) == list:
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer
                
            cores.append(get_var_wrap('core_%d' % (i + 1),
                                      shape=[out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]],
                                      initializer=cinit,
                                      regularizer=creg,
                                      trainable=trainable,
                                      cpu_variable=cpu_variables))                                                    
       
        full = filters
        
        for i in range(d):            
            full = tf.reshape(full, [-1, ranks[i]])
            core = tf.transpose(cores[i], [1, 0])
            core = tf.reshape(core, [ranks[i], -1])
            full = tf.matmul(full, core)
            
        out_ch = np.prod(out_ch_modes)
        inp_ch = np.prod(inp_ch_modes)
        
        fshape = [window[0], window[1], window[2]]
        order = [0, 1,2]
        inord = []
        outord = []
        for i in range(d):
            fshape.append(inp_ch_modes[i])
            inord.append(2 + 2 * i+1)
            fshape.append(out_ch_modes[i])
            outord.append(2 + 2 * i + 2)
        order += inord + outord
        full = tf.reshape(full, fshape)
        full = tf.transpose(full, order)
        full = tf.reshape(full, [window[0], window[1], window[2], inp_ch, out_ch])

        tmp = tf.nn.conv3d(inp, full, strides, padding,name='conv3d_tensor')


        

        
        if biases_initializer is not None:
            biases = get_var_wrap('biases',
                                  shape=[out_ch],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
            
            out = tf.add(tmp, biases, name='out')
        else:
            out = tf.identity(tmp, name='out')

    return out
