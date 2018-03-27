#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import model_multi_gpu_bis
import time
from tensorflow.python.client import device_lib
import time

from model_multi_gpu_bis import loss_CE,loss_IOU
from model_multi_gpu_bis import average_gradients

h = 512   #4032
w = 512   #3024
c_image = 3
c_label = 1



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default = './data_image.csv')

parser.add_argument('--test_dir',
                    default = './data_test.csv')

parser.add_argument('--model_dir',
                    default = './gpu_model')

parser.add_argument('--epochs',
                    type = int,
                    default = 100)

parser.add_argument('--peochs_per_eval',
                    type = int,
                    default = 1)

parser.add_argument('--logdir',
                    default = './gpu_logs')

parser.add_argument('--batch_size',
                    type = int,
                    default = 4)

parser.add_argument('--learning_rate',
                    type = float,
                    default = 1e-3)

#衰减系数
parser.add_argument('--decay_rate',
                    type = float,
                    default = 0.9)

parser.add_argument('--moving_average_rate',
                    type = float,
                    default = 0.99)

#衰减速度
parser.add_argument('--decay_step',
                    type = int,
                    default = 150)

parser.add_argument('--num_train',
                    type = int,
                    default = 10)

parser.add_argument('--weight',
                    nargs = '+',
                    type = float,
                    default = [1,1])

parser.add_argument('--random_seed',
                    type = int,
                    default = 1234)


flags = parser.parse_args()
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

def data_augmentation(image,label,training=True):
    if training:
        image_label = tf.concat([image,label],axis = -1)

        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)


        image = maybe_flipped[:, :, :-1]
        mask = maybe_flipped[:, :, -1:]

        image = tf.image.random_brightness(image, 0.7)
        #image = tf.image.random_hue(image, 0.3)
        #设置随机的对比度
        #tf.image.random_contrast(image,lower=0.3,upper=1.0)

        return image, mask

def get_input(queue,augmentation=True):
    csv_reader = tf.TextLineReader(skip_header_lines=1)

    _, csv_content = csv_reader.read(queue)

    image_path, label_path = tf.decode_csv(csv_content,record_defaults=[[""],[""]])

    image_file = tf.read_file(image_path)
    label_file = tf.read_file(label_path)

    image = tf.image.decode_jpeg(image_file, channels = 3)
    image.set_shape([h,w,c_image])
    image = tf.cast(image, tf.float32)

    label = tf.image.decode_jpeg(label_file, channels = 1)
    label.set_shape([h,w,c_label])

    label = tf.cast(label,tf.float32)
    label = label / (tf.reduce_max(label) + 1e-7)


    #数据增强
    if augmentation:
        image,label = data_augmentation(image,label)
    else:
        pass
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*flags.batch_size
    image, label = tf.train.shuffle_batch([image, label], batch_size=flags.batch_size, capacity=capacity,
                           min_after_dequeue=min_after_dequeue)
    return image,label
    #return tf.train.shuffle_batch([image,label],batch_size=flags.batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_loss(x,y_,is_training,scope,reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse_variables):
        y = model_multi_gpu_bis.dss_model(x,is_training=is_training)
    cross_entropy = loss_CE(y,y_)
    regularization_loss = tf.add_n(tf.get_collection('losses', scope))
    loss = cross_entropy + regularization_loss
    return loss,y


def main(flags):
    #简单的计算放在cpu，只有神经网络的训练过程放在gpu
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        current_time = time.strftime("%m/%d/%H/%M/%S")
        train_logdir = os.path.join(flags.logdir, "image", current_time)
        test_logdir = os.path.join(flags.logdir, "test", current_time)
        gpus = len(get_available_gpus())
        train = pd.read_csv(flags.data_dir)
        test = pd.read_csv(flags.test_dir)

        num_train = train.shape[0]
        num_test = test.shape[0]

        #获取训练batch
        train_csv = tf.train.string_input_producer([flags.data_dir])
        test_csv = tf.train.string_input_producer([flags.test_dir])
        X,y_ = get_input(train_csv,augmentation=True)
        test_image,test_label = get_input(test_csv,augmentation=False)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [X, y_], capacity=2 * gpus)

        X_test_batch_op, y_test_batch_op = tf.train.batch([test_image, test_label], batch_size=flags.batch_size,
                                                          capacity=flags.batch_size * 2, allow_smaller_final_batch=True)

        #regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        #定义训练轮数今儿指数衰减的学习率
        '''global_step = tf.get_variable(
            'global_step',[],initializer=tf.constant_initializer(0),
            trainable=False)'''
        global_step = tf.Variable(0,dtype=tf.int64,trainable=False,name='global_step')
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                                   decay_steps=flags.decay_step,
                                                   decay_rate=flags.decay_rate)

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('GPU_%d' % i) as scope:

                        #mode = tf.placeholder(tf.bool, name='mode')
                        mode = tf.constant(True,dtype=tf.bool)

                        #print(mode)

                        #image_batch, label_batch = batch_queue.dequeue()
                        #pred = model_multi_gpu_bis.unet(image_batch,mode)

                        '''cross_entropy = loss_CE(pred, y_)
                        regularization_loss = tf.add_n(tf.get_collection('losses',scope))

                        loss = cross_entropy + regularization_loss

                        cur_loss = loss'''
                        cur_loss,pred = get_loss(X,y_,is_training=mode,scope=scope)

                        pred_bis = tf.nn.sigmoid(pred)

                        tf.add_to_collection("inputs", X)
                        tf.add_to_collection('outputs', pred_bis)

                        tf.summary.image('Input image/gpu_%s'%i, X)
                        tf.summary.histogram('Predicted Mask/gpu_%s'%i, pred_bis)
                        tf.summary.image('Predicted Mask/gpu_%s'%i, pred_bis)
                        tf.summary.image('Label/gpu_%s'%i, y_)

                        tf.summary.scalar('Cross entropy loss/gpu_%s'%i,cur_loss)
                        tf.summary.scalar("Learning_rate", learning_rate)
                        #在第一次声明变量之后，将控制变量重用的参数设置为True，这样可以让不同的GPU更新同一组参数，tf,name_scope
                        #函数并不会影响tf.get_variable的命名空间
                        #原因是使用Adam或者RMSProp优化函数时，Adam函数会创建一个Adam变量，目的是保存你使用tensorflow创建的graph
                        # 中的每个可训练参数的动量，但是这个Adam是在reuse=True条件下创建的，之后reuse就回不到None或者False上去，
                        # 当reuse=True，就会在你当前的scope中reuse变量，如果在此scope中进行优化操作，就是使用AdamOptimizer等，
                        # 他就会重用slot variable，这样子会导致找不到Adam变量，进而报错
                        tf.get_variable_scope().reuse_variables()

                        #使用当前的GPU计算所有变量的梯度
                        grads = opt.compute_gradients(cur_loss)
                        tower_grads.append(grads)



        #计算变量的平均梯度，并输出到tensorboard日志中
        grads = average_gradients(tower_grads)
        for grad,var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name,grad)
                #tf.summary.scalar('gradients_on_average/%s' % var.op.name,grad)

        apply_gradient_op = opt.apply_gradients(
            grads,global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
            #tf.summary.scalar(var.op.name, var)

        #计算变量的滑动梯度
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        train_op = tf.group(apply_gradient_op,variables_averages_op)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_writer = tf.summary.FileWriter(test_logdir)

            if os.path.exists(flags.model_dir) and tf.train.checkpoint_exists(flags.model_dir):
                latest_check_point = tf.train.latest_checkpoint(flags.model_dir)
                saver.restore(sess,latest_check_point)

            else:
                try:
                    os.rmdir(flags.model_dir)

                except OSError:
                    pass

                os.mkdir(flags.model_dir)
            try:
                for epoch in range(flags.epochs):

                    for step in range(0,num_train,flags.batch_size):
                        start_time = time.time()
                        _,loss_value,step_value = sess.run([train_op,cur_loss,global_step])
                        duration = time.time() - start_time

                        summary = sess.run(summary_op)
                        train_writer.add_summary(summary, step_value)

                        num_examples_per_step = flags.batch_size * gpus

                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / gpus

                        format_str = ('epoch % d, step % d, loss = %.5f (%.1f examples/ )'
                                      ' sec; %.5f sec/batch')
                        print(format_str % (epoch+1,step,loss_value,examples_per_sec,sec_per_batch))

                    '''for step in range(0,num_test,flags.batch_size):
                        x = tf.placeholder(tf.float32, shape=[None, h, w, c_image], name='X')
                        y = tf.placeholder(tf.float32, shape=[None, h, w, c_label], name='y')
                        mode = tf.placeholder(tf.bool, name='mode')

                        X_test, y_test = sess.run([X_test_batch_op, y_test_batch_op])
                        step_ce, step_summary = sess.run([cur_loss, summary_op],
                                                         feed_dict={x: X_test, y: y_test, mode: False})

                        test_writer.add_summary(step_summary, epoch * (
                                    num_train // flags.batch_size) + step // flags.batch_size * num_train // num_test)
                        print('Test loss_CE:{}'.format(step_ce))'''

            finally:
                coord.request_stop()
                coord.join(threads)
                saver.save(sess,'{}/model.ckpt'.format(flags.model_dir))

if __name__ == '__main__':
    main(flags)