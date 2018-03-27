#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import model_multi_gpu_bis
import time
from model_multi_gpu_bis import train_op
from model_multi_gpu_bis import loss_IOU
from model_multi_gpu_bis import loss_CE


h = 512   #4032
w = 512   #3024
c_image = 3
c_label = 1
image_mean = [141.23,128.69,119.92]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default = './data_image.csv')

parser.add_argument('--test_dir',
                    default = './data_test.csv')

parser.add_argument('--model_dir',
                    default = './model1')

parser.add_argument('--epochs',
                    type = int,
                    default = 20)

parser.add_argument('--peochs_per_eval',
                    type = int,
                    default = 1)

parser.add_argument('--logdir',
                    default = './logs1')

parser.add_argument('--batch_size',
                    type = int,
                    default = 4)

parser.add_argument('--is_cross_entropy',
                    action = 'store_false')

parser.add_argument('--learning_rate',
                    type = float,
                    default = 1e-3)
#衰减系数
parser.add_argument('--decay_rate',
                    type = float,
                    default = 0.9)

#衰减速度
parser.add_argument('--decay_step',
                    type = int,
                    default = 1)

parser.add_argument('--weight',
                    nargs = '+',
                    type = float,
                    default = [1,1])

parser.add_argument('--random_seed',
                    type = int,
                    default = 1234)

parser.add_argument('--gpu',
                    type = str,
                    default = 1)

flags = parser.parse_args()



def set_config():

    ''''#允许增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    '''

    #控制使用率
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
    config = tf.ConfigProto(gpu_options = gpu_options)
    session = tf.Session(config=config)

def data_augmentation(image,label,training=True):
    if training:
        image_label = tf.concat([image,label],axis = -1)

        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(image_label)


        image = maybe_flipped[:, :, :-1]
        mask = maybe_flipped[:, :, -1:]

        image = tf.image.random_brightness(image, 0.7)
        #image = tf.image.random_hue(image, 0.3)
        #设置随机的对比度
        #tf.image.random_contrast(image,lower=0.3,upper=1.0)

        return image, mask

def read_csv(queue,augmentation=True):
    #csv = tf.train.string_input_producer(['./data/train/csv','./data/test.csv'])
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
    return image,label

def main(flags):
    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(flags.logdir, "image", current_time)
    test_logdir = os.path.join(flags.logdir, "test", current_time)

    train = pd.read_csv(flags.data_dir)


    num_train = train.shape[0]

    test = pd.read_csv(flags.test_dir)
    num_test = test.shape[0]

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = [flags.batch_size,h,w,c_image],name = 'X')
    y = tf.placeholder(tf.float32,shape = [flags.batch_size,h,w,c_label], name = 'y')
    mode = tf.placeholder(tf.bool, name='mode')

    pred = model_multi_gpu_bis.unet(input_image=X,is_training=mode)

    if flags.is_cross_entropy:
        loss = loss_CE(pred,y)
        CE_op = loss_CE(pred, y)
        tf.summary.scalar("CE", CE_op)

    else:
        loss = -loss_IOU(pred,y)
        IOU_op = loss_IOU(pred, y)
        tf.summary.scalar('IOU:', IOU_op)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                               tf.cast(num_train / flags.batch_size * flags.decay_step, tf.int32),
                                               flags.decay_rate, staircase=True)

    with tf.control_dependencies(update_ops):
        training_op = train_op(loss,learning_rate)


    train_csv = tf.train.string_input_producer(['data_image.csv'])
    test_csv = tf.train.string_input_producer(['data_test.csv'])

    train_image, train_label = read_csv(train_csv,augmentation=True)
    test_image, test_label = read_csv(test_csv,augmentation=False)

    #batch_size是返回的一个batch样本集的样本个数。capacity是队列中的容量
    X_train_batch_op, y_train_batch_op = tf.train.shuffle_batch([train_image, train_label],batch_size = flags.batch_size,
                                              capacity = flags.batch_size*5,min_after_dequeue = flags.batch_size*2,
                                              allow_smaller_final_batch = True)

    X_test_batch_op, y_test_batch_op = tf.train.batch([test_image, test_label],batch_size = flags.batch_size,
                                                        capacity = flags.batch_size*2,allow_smaller_final_batch = True)



    print('Shuffle batch done')
    #tf.summary.scalar('loss/Cross_entropy', CE_op)
    pred_bis = tf.nn.sigmoid(pred)

    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', mode)
    tf.add_to_collection('pred', pred_bis)

    tf.summary.image('Input Image:', X)
    tf.summary.image('Label:', y)
    tf.summary.image('Predicted Image:', pred_bis)

    tf.summary.scalar("learning_rate", learning_rate)

    # 添加任意shape的Tensor，统计这个Tensor的取值分布
    tf.summary.histogram('Predicted Image:', pred_bis)


    #添加一个操作，代表执行所有summary操作，这样可以避免人工执行每一个summary op
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)


        saver = tf.train.Saver()
        if os.path.exists(flags.model_dir) and tf.train.checkpoint_exists(flags.model_dir):
            latest_check_point = tf.train.latest_checkpoint(flags.model_dir)
            saver.restore(sess, latest_check_point)

        else:
            print('No model')
            try:
                os.rmdir(flags.model_dir)
            except Exception as e:
                print(e)
            os.mkdir(flags.model_dir)

        try:
            #global_step = tf.train.get_global_step(sess.graph)

            #使用tf.train.string_input_producer(epoch_size, shuffle=False),会默认将QueueRunner添加到全局图中，
            #我们必须使用tf.train.start_queue_runners(sess=sess)，去启动该线程。要在session当中将该线程开启,不然就会挂起。然后使用coord= tf.train.Coordinator()去做一些线程的同步工作,
            #否则会出现运行到sess.run一直卡住不动的情况。
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(flags.epochs):
                for step in range(0,num_train,flags.batch_size):
                    X_train, y_train = sess.run([X_train_batch_op,y_train_batch_op])

                    if flags.is_cross_entropy:

                        _,step_ce,step_summary,global_step_value = sess.run([training_op,CE_op,summary_op,global_step],feed_dict={X:X_train,y:y_train,mode:True})

                        train_writer.add_summary(step_summary,global_step_value)
                        print('epoch:{} step:{} loss_CE:{}'.format(epoch+1, global_step_value, step_ce))

                    else:
                        _,step_iou,step_summary,global_step_value = sess.run([training_op, IOU_op, summary_op, global_step],feed_dict={X: X_train, y: y_train, mode: True})

                        train_writer.add_summary(step_summary, global_step_value)
                        print('epoch:{} step:{} loss_IOU:{}'.format(epoch + 1, global_step_value, step_iou))

                for step in range(0,num_test,flags.batch_size):
                    if flags.is_cross_entropy:
                        X_test, y_test = sess.run([X_test_batch_op,y_test_batch_op])
                        step_ce,step_summary = sess.run([CE_op,summary_op],feed_dict={X: X_test,y:y_test,mode:False})

                        test_writer.add_summary(step_summary,epoch * (num_train // flags.batch_size) + step // flags.batch_size * num_train // num_test)
                        print('Test loss_CE:{}'.format(step_ce))

                    else:
                        X_test, y_test = sess.run([X_test_batch_op, y_test_batch_op])
                        step_iou,step_summary = sess.run([IOU_op,summary_op],feed_dict={X: X_test, y: y_test, mode: False})

                        test_writer.add_summary(step_summary,epoch * (num_train // flags.batch_size) + step // flags.batch_size * num_train // num_test)
                        print('Test loss_IOU:{}'.format(step_iou))

            saver.save(sess, '{}/model.ckpt'.format(flags.model_dir))

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    set_config()
    main(flags)















