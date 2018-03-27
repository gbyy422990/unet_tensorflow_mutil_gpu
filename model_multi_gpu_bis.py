#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import numpy as np
import re
import math

loss_weight = np.array([1,1])
TOWER_NAME = 'tower'
num_class =2

g_mean = [141.23,128.69,119.92]

def _variable_on_cpu(name,shape,initializer,use_fp16=False):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)

    return var

def _variable_with_weight_decay(name,shape,initializer,wd):
    var =_variable_on_cpu(name,shape,initializer)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)

    return var

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)



def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def upsampling_2d(tensor,name,size=(2,2)):
    h_,w_,_ = tensor.get_shape().as_list()[1:]
    h_multi,w_multi = size
    h = h_multi * h_

    w = w_multi * w_
    target = tf.image.resize_nearest_neighbor(tensor,size=(h,w),name='upsample_{}'.format(name))
    #print('shape',target.get_shape())

    return target

def upsampling_concat(input_A,input_B,name):
    upsampling = upsampling_2d(input_A,name=name,size=(2,2))
    up_concat = tf.concat([upsampling,input_B],axis=-1,name='up_concat_{}'.format(name))
    #print('concat',up_concat.get_shape())

    return up_concat

'''
def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


'''


def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1,1],
           padding='SAME',
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None
           ):
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             initializer=msra_initializer(kernel_h,num_in_channels),
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            #outputs = batch_norm_layer(outputs,is_training,scope=sc.name)
            outputs = tf.layers.batch_normalization(outputs, training=is_training, name='bn_{}'.format(scope))
            #outputs = batch_norm_for_conv2d(outputs, is_training,
             #                               bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        _activation_summary(outputs)

    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):

    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)

        return outputs

def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             initializer=orthogonal_initializer(),
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            #outputs = batch_norm_layer(outputs, is_training, scope=sc.name)
            outputs = tf.layers.batch_normalization(outputs,training=is_training,name='bn_{}'.format(scope))
            #outputs = batch_norm_for_conv2d(outputs, is_training,
             #                               bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        #_activation_summary(outputs)

        return outputs

def dss_model(input_image,is_training,bn_decay=None):

    #conv1
    input_image = input_image/127.5 - 1

    #norm1 = tf.nn.lrn(input_image, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
     #                 name='norm1')

    #conv0 = conv2d(input_image,3,[1,1],scope='conv0',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv1 = conv2d(input_image,64,[3,3],scope='conv1',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv1_bis = conv2d(conv1,64,[3,3],scope='conv1_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    pool1 = max_pool2d(conv1_bis,kernel_size=[2,2],scope='pool1',stride=[2,2],padding='VALID')

    #conv2
    conv2 = conv2d(pool1,128,[3,3],scope='conv2',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv2_bis = conv2d(conv2,128,[3,3],scope='conv2_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    pool2 = max_pool2d(conv2_bis,kernel_size=[2,2],scope='pool2',stride=[2,2],padding='VALID')

    #conv3
    conv3 = conv2d(pool2,256,[3,3],scope='conv3',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv3_bis = conv2d(conv3,256,[3,3],scope='conv3_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    pool3 = max_pool2d(conv3_bis,kernel_size=[2,2],scope='pool3',stride=[2,2],padding='VALID')

    #conv4
    conv4 = conv2d(pool3,512,[3,3],scope='conv4',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv4_bis = conv2d(conv4,512,[3,3],scope='conv4_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    pool4 = max_pool2d(conv4_bis,kernel_size=[2,2],scope='pool4',stride=[2,2],padding='VALID')

    #conv5
    conv5 = conv2d(pool4,1024,[3,3],scope='conv5',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv5_bis = conv2d(conv5,1024,[3,3],scope='conv5_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    #upsampling1
    #up6 = upsampling_concat(conv5_bis,conv4_bis,name='up6')
    up6 = conv2d_transpose(conv5_bis,512,kernel_size=[3,3],scope='up6',stride=[2,2],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)
    up6 = tf.concat([up6,conv4_bis],axis=-1,name='up_concat_6')
    conv6 = conv2d(up6,512,[3,3],scope='conv6',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv6_bis = conv2d(conv6,512,[3,3],scope='conv6_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    #upsampling2
    #up7 = upsampling_concat(conv6_bis,conv3_bis,name='up7')
    up7 = conv2d_transpose(conv6_bis,256,kernel_size=[3,3],scope='up7',stride=[2,2],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)
    up7 = tf.concat([up7,conv3_bis],axis=-1,name='up_concat_7')
    conv7 = conv2d(up7,256,[3,3],scope='conv7',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv7_bis = conv2d(conv7,256,[3,3],scope='conv7_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    #upsampling3
    #up8 = upsampling_concat(conv7_bis,conv2_bis,name='up8')
    up8 = conv2d_transpose(conv7_bis,128,kernel_size=[3,3],scope='up8',stride=[2,2],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)
    up8 = tf.concat([up8,conv2_bis],axis=-1,name='up_concat_8')
    conv8 = conv2d(up8,128,[3,3],scope='conv8',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv8_bis = conv2d(conv8,128,[3,3],scope='conv8_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    #upsampling4
    #up9 = upsampling_concat(conv8_bis,conv1_bis,name='up9')
    up9 = conv2d_transpose(conv8_bis,64,kernel_size=[3,3],scope='up9',stride=[2,2],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)
    up9 = tf.concat([up9,conv1_bis],axis=-1,name='up_concat_9')
    conv9 = conv2d(up9,64,[3,3],scope='cpnv9',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay)
    conv9_bis = conv2d(conv9,64,[3,3],scope='conv9_bis',stride=[1,1],padding='SAME',bn=True,is_training=is_training,bn_decay=bn_decay)

    #finally
    conv10 = conv2d(conv9_bis,1,[1,1],scope='conv10_output',stride=[1,1],padding='SAME',bn=False,is_training=is_training,bn_decay=bn_decay,activation_fn=None)

    return conv10


#交叉熵损失+权重
def loss_CE(y_pred,y_true):
    '''flat_logits = tf.reshape(y_pred,[-1,num_class])
    flat_labels = tf.reshape(y_true,[-1,num_class])
    class_weights = tf.constant(loss_weight,dtype=np.float32)
    weight_map = tf.multiply(flat_labels,class_weights)
    weight_map = tf.reduce_sum(weight_map,axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels,logits=flat_logits)

    weighted_loss = tf.multiply(loss_map,weight_map)

    cross_entropy_mean = tf.reduce_mean(weighted_loss)'''

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

    return cross_entropy_mean

#IOU损失
def loss_IOU(y_pred,y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    flat_logits = tf.reshape(y_pred, [-1, H * W])
    flat_labels = tf.reshape(y_true, [-1, H * W])
    intersection = 2 * tf.reduce_sum(flat_logits * flat_labels, axis=1) + 1e-7
    denominator = tf.reduce_sum(flat_logits, axis=1) + tf.reduce_sum(flat_labels, axis=1) + 1e-7
    iou = tf.reduce_mean(intersection / denominator)

    return iou

def train_op(loss,learning_rate):

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss,global_step=global_step)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        #Note that each grad_vars looks like the following:
        #((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g,_ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)

            #Append on a 'tower' dimension which we will average over below
            grads.append(expanded_g)

        #Average over the 'tower' dimension
        grad = tf.concat(axis=0,values=grads)
        grad = tf.reduce_mean(grad,0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad,v)
        average_grads.append(grad_and_var)

    return average_grads


tf.ConfigProto()