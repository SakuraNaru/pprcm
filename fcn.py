import copy
import heapq
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.training import moving_averages

import sys
from sklearn.model_selection import train_test_split
from tensorflow.python.ops import array_ops
import utils
from tensorflow import set_random_seed
set_random_seed(1)

EARLY_STOP_PATIENCE=5


def readData(dataset_name):
    x = []
    # print(np.shape(x))
    y = []
    label_name = []
    lists = os.listdir('%s/data/input' % dataset_name)

    for i in range(0, len(lists)):
        x = np.concatenate((x, np.load('%s/data/input/input_x_%s.npy' % (dataset_name, i),allow_pickle=True)), axis=0)
    for i in range(0, len(lists)):
        y = np.concatenate((y, np.load('%s/data/output/output_y_%s.npy' % (dataset_name, i),allow_pickle=True)), axis=0)

    print(np.shape(x)[0])
    return x, y


def singleLayer(input,w_size,h_size,c_size,layer_no,keep_probability):
    W = utils.weight_variable([w_size, h_size, input.get_shape()[3].value, c_size], name='W%s'%layer_no)
    b = utils.bias_variable([c_size], 'bias%s'%layer_no)
    conv = utils.conv2d(input, W, b,name='conv_%s'%layer_no)
    # conv1 = tf.layers.batch_normalization(conv1, training=is_train)

    conv=tf.nn.dropout(conv,keep_prob=keep_probability)
    output = tf.nn.leaky_relu(conv,name='relu_%s'%layer_no)
    # output=GroupNorm(output,16)

    return output

def fcnLayer(x,keep_probability):
    w_size=3
    h_size=3

    layer_size=[128,128,64,64,64,64,64,64,64]

    # for i in range(len(layer_size)):
    x1=singleLayer(x,w_size,h_size,layer_size[0],1,keep_probability)
    x2=singleLayer(x1,w_size,h_size,layer_size[1],2,keep_probability)
    x3=singleLayer(x2,w_size,h_size,layer_size[2],3,keep_probability)
    x4=singleLayer(x3,w_size,h_size,layer_size[3],4,keep_probability)
    x5=singleLayer(x4,w_size,h_size,layer_size[4],5,keep_probability)

    # x6=singleLayer(x5,w_size,h_size,layer_size[5],6,keep_probability)
    # x7=singleLayer(x6,w_size,h_size,layer_size[6],7,keep_probability)
    # x8=singleLayer(x7,1,1,layer_size[7],8,keep_probability)
    # x9=singleLayer(x8,1,1,layer_size[8],9,keep_probability)


    # W_out = utils.weight_variable([1, 1, x6.get_shape()[3].value, 1], name='W_out')
    # b_out = utils.bias_variable([1], 'bias_out')
    # output = utils.conv2d(x6, W_out, b_out)
    #
    # W_out = utils.weight_variable([1, 1, x6.get_shape()[3].value, 1], name='W_out')
    # b_out = utils.bias_variable([1], 'bias_out')
    # output = utils.conv2d(x6, W_out, b_out)
    #
    W_out = utils.weight_variable([1, 1, x5.get_shape()[3].value, 1], name='W_out')
    b_out = utils.bias_variable([1], 'bias_out')
    output = utils.conv2d(x5, W_out, b_out,name='conv_out')
    # packet=[conv1,conv2,conv3,conv4,output]
    return output

def convLayer(x,layer):
    W = utils.weight_variable([3, 3, x.get_shape()[3].value, 64], name='W%s'%layer)
    b = utils.bias_variable([64], 'bias%s'%layer)
    conv = utils.conv2d(x, W, b)
    conv = tf.nn.relu(conv)

    return conv
def resnet(x,is_train):
    W1 = utils.weight_variable([1, 1, x.get_shape()[3].value, 64], name='W1')
    b1 = utils.bias_variable([64], 'bias1')
    conv1 = utils.conv2d(x, W1, b1)
    # conv1 =tf.layers.batch_normalization(conv1,training=is_train)
    conv1 = tf.nn.relu(conv1)

    bn1=bottleneck(conv1,1,is_train)
    cc1=tf.concat([bn1,conv1],axis=3,name='C1')
    cc1=tf.nn.relu(cc1)
    bn2=bottleneck(cc1,2,is_train)
    cc2=tf.concat([bn2,cc1],axis=3,name='C2')
    cc2=tf.nn.relu(cc2)
    bn3 = bottleneck(cc2, 3,is_train)
    cc3 = tf.concat([bn3, cc2], axis=3, name='C2')
    cc3 = tf.nn.relu(cc3)

    W2 = utils.weight_variable([1, 1, cc3.get_shape()[3].value, 64], name='W2')
    b2 = utils.bias_variable([64], 'bias2')
    conv2 = utils.conv2d(cc3, W2, b2)
    # conv2 = tf.layers.batch_normalization(conv2, training=is_train)
    conv2 = tf.nn.relu(conv2)

    W3 = utils.weight_variable([1, 1, conv2.get_shape()[3].value, 1], name='W3')
    b3 = utils.bias_variable([1], 'bias3')
    conv3 = utils.conv2d(conv2, W3, b3)

    return conv3
def bottleneck(x,layer,is_train):
    W1=utils.weight_variable([1,1,x.get_shape()[3].value,64],name='W%s_1'%layer)
    conv1=utils.resConv2d(x,W1)
    # conv1 = tf.layers.batch_normalization(conv1, training=is_train)
    conv1=tf.nn.relu(conv1)

    W2 = utils.weight_variable([3, 3, conv1.get_shape()[3].value, 64], name='W%s_2' % layer)
    conv2 = utils.resConv2d(conv1, W2)
    # conv2 = tf.layers.batch_normalization(conv2, training=is_train)
    conv2 = tf.nn.relu(conv2)

    W3 = utils.weight_variable([1, 1, conv2.get_shape()[3].value, 64], name='W%s_3' % layer)
    conv3 = utils.resConv2d(conv2, W3)
    # conv3 = tf.layers.batch_normalization(conv3, training=is_train)

    return conv3

def fcnLayerNew(x,keep_probability):
    W1=utils.weight_variable([3,3,x.get_shape()[3].value,128],name='W1')
    b1=utils.bias_variable([128],'bias1')
    conv1_1=utils.conv2d(x,W1,b1,'conv_1')
    conv1_2=tf.nn.relu(conv1_1)

    W2 = utils.weight_variable([ 3, 3,conv1_2.get_shape()[3].value,128],name= 'W2')
    b2 = utils.bias_variable([128], 'bias2')
    conv2_1 = utils.conv2d(conv1_2, W2, b2,'conv_2')
    conv2_2 = tf.nn.relu(conv2_1,'relu_2')

    W3 = utils.weight_variable([ 3,3, conv2_2.get_shape()[3].value,64], name='W3')
    b3 = utils.bias_variable([64], 'bias3')
    conv3_1 = utils.conv2d(conv2_2, W3, b3,'conv_3')
    conv3_2 = tf.nn.relu(conv3_1,'relu_3')
    conv3_3 = utils.max_pool(conv3_2,2,name='conv_p3')

    W4 = utils.weight_variable([3, 3, conv3_3.get_shape()[3].value, 64], name='W4')
    b4 = utils.bias_variable([64], 'bias4')
    conv4_1 = utils.conv2d(conv3_3, W4, b4, 'conv_4')
    conv4_2 = tf.nn.relu(conv4_1, 'relu_4')
    conv4_3 = utils.max_pool(conv4_2, 2, name='conv_p4')

    W5 = utils.weight_variable([3, 3, conv4_3.get_shape()[3].value, 64], name='W5')
    b5 = utils.bias_variable([64], 'bias5')
    conv5_1 = utils.conv2d(conv4_3, W5, b5, 'conv_5')
    conv5_2 = tf.nn.relu(conv5_1, 'relu_5')
    conv5_3 = utils.max_pool(conv5_2, 2, name='conv_p5')

    # W6 = utils.weight_variable([1, 1, conv5_3.get_shape()[3].value, 1], name='W6')
    # b6 = utils.bias_variable([1], 'bias6')
    # conv6_1 = utils.conv2d(conv5_3, W6, b6, 'conv_6')
    # conv6_2 = tf.nn.relu(conv6_1, 'relu_6')

    # upscale
    deconv_shape1 = conv4_3.get_shape()
    W_t1=utils.weight_variable([4,4,deconv_shape1[3].value,64],name='W_t1')
    b_t1=utils.bias_variable([deconv_shape1[3].value],name='b_t1')
    conv_t1=utils.conv2d_transpose(conv5_3,W_t1,b_t1,tf.shape(conv4_3),stride=2)
    # fuse_1=tf.add(conv_t1,conv4_3,name='fuse_1')

    deconv_shape2 = conv3_3.get_shape()
    W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name='W_t2')
    b_t2 = utils.bias_variable([deconv_shape2[3].value], name='b_t2')
    conv_t2 = utils.conv2d_transpose(conv_t1, W_t2, b_t2, tf.shape(conv3_3), stride=2)
    # fuse_2 = tf.add(conv_t2, conv3_3, name='fuse_2')

    # deconv_shape1=conv5_3.get_shape()
    shape = tf.shape(x)
    deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 1])
    W_t3 = utils.weight_variable([4, 4, 1, deconv_shape1[3].value], name='W_t3')
    b_t3 = utils.bias_variable([1], name='b_t3')
    conv_t3 = utils.conv2d_transpose(conv_t2, W_t3, b_t3, deconv_shape3, stride=2,name='out')



    return conv_t3
def UNet(x,keep_probability):
    layer1=conv_block(x,1)
    pool1 = utils.max_pool(layer1, 2)

    layer2=conv_block(pool1,2)
    pool2=utils.max_pool(layer2,2)

    layer3=conv_block(pool2,3)
    pool3=utils.max_pool(layer3,2)

    layer4=conv_block(pool3,4)
    #upsampling

    up5=upsampling_bolck(layer3,layer4,5)
    layer5=conv_block(up5,5)

    up6=upsampling_bolck(layer2,layer5,6)
    layer6 = conv_block(up6, 6)

    up7 = upsampling_bolck(layer1, layer6, 7)
    layer7 = conv_block(up7, 7)

    W6 = utils.weight_variable([1, 1, layer7.get_shape()[3].value, 1], name='W6')
    b6 = utils.bias_variable([1], 'bias6')
    conv6_1 = utils.conv2d(layer7, W6, b6,'conv_6')

    return conv6_1

def conv_block(x,num):
    W1 = utils.weight_variable([3, 3, x.get_shape()[3].value, 128], name='W%s_1'%num)
    b1 = utils.bias_variable([128], 'bias%s_1'%num)
    conv1 = utils.conv2d(x, W1, b1,'conv_%s'%num)
    conv1 = tf.nn.relu(conv1)

    # W2 = utils.weight_variable([1, 1, conv1.get_shape()[3].value, 32], name='W%s_2'%num)
    # b2 = utils.bias_variable([32], 'bias%s_2'%num)
    # conv2 = utils.conv2d(conv1, W2, b2,'conv_%s'%num)
    # conv2 = tf.nn.relu(conv2)

    return conv1
def upsampling_bolck(x1,x2,num):
    shape = tf.shape(x1)
    outputs_shape = [shape[0], shape[1], shape[2], x1.get_shape()[3].value]
    W_t1 = utils.weight_variable([4, 4, x1.get_shape()[3].value, x2.get_shape()[3].value], name='W%s_t1'%num)
    b_t1 = utils.bias_variable([x1.get_shape()[3].value], name='b%s_t1'%num)
    conv_t1 = utils.conv2d_transpose(x2, W_t1, b_t1, outputs_shape, stride=2)
    fuse_1 = tf.concat([conv_t1, x1], axis=3,name='fuse%s_1'%num)

    return fuse_1

def deeplab(x):
    W1 = utils.weight_variable([3, 3, x.get_shape()[3].value, 128], name='W1')
    b1 = utils.bias_variable([128], 'bias1')
    conv1_1 = utils.conv2d(x, W1, b1)
    conv1_2 = tf.nn.relu(conv1_1)
    conv1_3=utils.max_pool(conv1_2,2)

    W2 = utils.weight_variable([3, 3, conv1_3.get_shape()[3].value, 128], name='W2')
    b2 = utils.bias_variable([128], 'bias2')
    conv2_1 = utils.conv2d(conv1_3, W2, b2)
    conv2_2 = tf.nn.relu(conv2_1)
    conv2_3 = utils.max_pool(conv2_2, 2)

    W3 = utils.weight_variable([3, 3, conv2_3.get_shape()[3].value, 128], name='W3')
    b3 = utils.bias_variable([128], 'bias3')
    conv3_1 = utils.atrous_conv2d(conv2_3, W2, b2)
    conv3_2 = tf.nn.relu(conv3_1)

    W4 = utils.weight_variable([3, 3, conv2_3.get_shape()[3].value, 128], name='W4')
    b4 = utils.bias_variable([128], 'bias4')
    conv4_1 = utils.atrous_conv2d(conv3_2, W4, b4)
    conv4_2 = tf.nn.relu(conv4_1)

    # fully connection

    W5 = utils.weight_variable([3, 3, conv4_2.get_shape()[3].value, 128], name='W5')
    b5 = utils.bias_variable([128], 'bias5')
    conv5_1 = utils.conv2d(conv4_2, W5, b5)
    conv5_2 = tf.nn.relu(conv5_1)

    # upscale
    inputs_shape = tf.shape(conv1_3)
    outputs_shape = [inputs_shape[0], inputs_shape[1], inputs_shape[2], conv1_3.get_shape()[3].value]
    W_t1 = utils.weight_variable([4, 4, conv1_3.get_shape()[3].value, 128], name='W_t1')
    b_t1 = utils.bias_variable([conv1_3.get_shape()[3].value], name='b_t1')
    conv_t1 = utils.conv2d_transpose(conv5_2, W_t1, b_t1, outputs_shape, stride=2)
    fuse_1 = tf.add(conv_t1, conv1_3, name='fuse_1')

    # inputs_shape = tf.shape(conv1_3)
    # outputs_shape = [inputs_shape[0], inputs_shape[1], inputs_shape[2], conv1_3.get_shape()[3].value]
    # W_t2 = utils.weight_variable([4, 4, conv1_3.get_shape()[3].value,fuse_1.get_shape()[3].value], name='W_t2')
    # b_t2 = utils.bias_variable([conv1_3.get_shape()[3].value], name='b_t2')
    # conv_t2 = utils.conv2d_transpose(fuse_1,W_t2, b_t2, outputs_shape,stride=2)
    # fuse_2=tf.add(conv_t2,conv1_1_3,name='fuse_2')

    inputs_shape = tf.shape(x)
    outputs_shape = [inputs_shape[0], inputs_shape[1], inputs_shape[2], 128]
    W_t3 = utils.weight_variable([4, 4, 128, fuse_1.get_shape()[3].value], name='W_t3')
    b_t3 = utils.bias_variable([128], name='b_t3')
    conv_t3 = utils.conv2d_transpose(fuse_1, W_t3, b_t3, outputs_shape, stride=2)

    # annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    # return annotation_pred, conv_t3

    W0 = utils.weight_variable([1, 1, conv_t3.get_shape()[3].value, 1], name='W0')
    b0 = utils.bias_variable([1], 'bias0')
    conv0_1 = utils.conv2d(conv_t3, W0, b0)
    return conv0_1
def train(loss_val, var_list,learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    # for grad, var in grads:
    #     utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def calmIoU(label,logits):
    # print(np.shape(label))
    # print(np.shape(logits))
    logits = np.squeeze(logits, axis=(0, 3))
    pos=np.where(logits>0.5)
    predict_matrix=np.zeros_like(logits)
    predict_matrix[pos]=1
    intersection=np.logical_and(predict_matrix,label)
    union=np.logical_or(predict_matrix,label)
    return np.sum(intersection)/np.sum(union)

def makeWeight(y_train):
    weight_list=[]
    for i in y_train:
        weight_list.append(np.size(i))
    weight_list=weight_list/np.sum(weight_list)
    return weight_list
def fcnModelTf(x_train,x_val,x_test,y_train,y_val,y_test):
    x = tf.placeholder(tf.float32, [None, None,None, 80],name='input')
    y = tf.placeholder(tf.float32, [None,None, None],name='output')
    epoch = tf.Variable(tf.constant(0))
    is_train = tf.placeholder(tf.bool,name='is_train')
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    output=fcnLayer(x,keep_probability)

    learning_rate = tf.train.exponential_decay(5e-4, epoch, 20000, 0.9, staircase=True)
    with tf.name_scope('loss'):
        loss_s = corss_matrix_focal_loss(labels=y[:, :, :, tf.newaxis], logits=output, alpha=0.9, gamma=2)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_s, global_step=epoch)



    sigmoid_out=tf.nn.sigmoid(output,name='sigmoid')
    predict = tf.round(sigmoid_out,name='predict')

    acc_s = accuracy_matrix(labels=y[:, :, :, tf.newaxis], logits=predict)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver(max_to_keep=1)
    if not os.path.exists('%s/model' % dataset_name):
        os.mkdir('%s/model' % dataset_name)
    if not os.path.exists('%s/log' % dataset_name):
        os.mkdir('%s/log' % dataset_name)
    kf_no = 0

    max_fscore=-1
    min_loss = sys.maxsize
    kf_no += 1
    patience = EARLY_STOP_PATIENCE
    test_precision=[]
    test_recall=[]
    test_fscore=[]
    test_loss=[]

    train_precision = []
    train_recall = []
    train_fscore = []
    train_loss = []

    merged = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(800):
            x_train,y_train=shuffle_data(x_train,y_train)
            length_train = len(x_train)
            sum_loss = 0
            sum_tp=0
            sum_tn=0
            sum_fp=0
            sum_fn=0
            for i in range(length_train):
                sigmoid_train,_,train_cost, lr, acc_mark= sess.run([sigmoid_out,optimizer,loss_s, learning_rate,acc_s],
                                       feed_dict={x: [x_train[i]], y: [y_train[i]], is_train: True, keep_probability:0.9})
                # print('****%s'%L2_data)
                true_positive, true_negative, false_positive, false_negative=acc_mark

                sum_tp += true_positive
                sum_tn += true_negative
                sum_fp += false_positive
                sum_fn += false_negative

                sum_loss += train_cost
            mcc=calMCC(sum_tp,sum_tn,sum_fp,sum_fn)

            sum_precision = 0 if (sum_tp + sum_fp) == 0 else sum_tp / (sum_tp + sum_fp)
            sum_recall = 0 if (sum_tp + sum_fn) == 0 else sum_tp / (sum_tp + sum_fn)

            sum_loss /= length_train
            sum_fscore=calFscore(sum_precision,sum_recall,1)

            train_precision.append(sum_precision)
            train_recall.append(sum_recall)
            train_fscore.append(sum_fscore)
            train_loss.append(sum_loss)

            print('loss:%s precision:%s recall:%s fscore:%s mcc%s' % (sum_loss, sum_precision, sum_recall,sum_fscore,mcc))
            #early stop
            if (step)%5==0:
                length_val = len(x_val)
                sum_tp = 0
                sum_tn = 0
                sum_fp = 0
                sum_fn = 0
                sum_val_loss = 0
                for i in range(length_val):
                    sigmoid_val,val_loss,pack= sess.run([sigmoid_out,loss_s,acc_s],feed_dict={x: [x_val[i]], y: [y_val[i]], is_train: False,keep_probability:1})
                    true_positive, true_negative, false_positive, false_negative=pack
                    # print('tp:%s tn:%s fp:%s fn:%s '%(true_positive, true_negative, false_positive, false_negative))
                    sum_tp += true_positive
                    sum_tn += true_negative
                    sum_fp += false_positive
                    sum_fn += false_negative

                    sum_val_loss += val_loss

                mcc = calMCC(sum_tp, sum_tn, sum_fp, sum_fn)
                sum_val_precision = 0 if (sum_tp + sum_fp) == 0 else sum_tp / (sum_tp + sum_fp)
                sum_val_recall = 0 if (sum_tp + sum_fn) == 0 else sum_tp / (sum_tp + sum_fn)

                sum_val_loss /= length_val
                sum_val_fscore = calFscore(sum_val_precision, sum_val_recall, 1)

                test_precision.append(sum_val_precision)
                test_recall.append(sum_val_recall)
                test_fscore.append(sum_val_fscore)
                test_loss.append(sum_val_loss)

                print('val loss:%s sum precision:%s recall:%s fscore:%s mcc:%s' % ( sum_val_loss,sum_val_precision, sum_val_recall,sum_val_fscore,mcc))
                if sum_val_fscore>(max_fscore):
                    max_fscore=sum_val_fscore
                    patience = EARLY_STOP_PATIENCE
                    print('success save!')
                    saver.save(sess, '%s/model/model.ckpt' % dataset_name)
                else:
                    patience-=1
                if patience==0:

                    print('early stop! mini loss is %s'%min_loss)
                    break

        np.savez('%s/model/train.npz' % dataset_name, train_precision, train_recall, train_fscore,train_loss)
        np.savez('%s/model/test.npz' % dataset_name, test_precision, test_recall, test_fscore,test_loss)

        precision_list = []
        recall_list = []
        logits_list = []
        num_list=[]

        conv1_3s=[]
        conv2_3s=[]
        conv3_3s=[]
        conv6_1s=[]
        conv_t1s=[]
        conv_t2s=[]
        conv2_1_3s=[]
        conv1_1_3s=[]

        sum_tp = 0
        sum_tn = 0
        sum_fp = 0
        sum_fn = 0
        sum_test_precision = 0
        sum_test_recall = 0
        # top_l=0
        length_test=len(x_test)

        for i in range(length_test):
            sigmoid_predict,pack= sess.run([sigmoid_out,acc_s], feed_dict={x: [x_test[i]], y: [y_test[i]], is_train: False,keep_probability:1})

            true_positive, true_negative, false_positive, false_negative=pack
            sum_tp+=true_positive
            sum_tn += true_negative
            sum_fp += false_positive
            sum_fn += false_negative


            precision=0 if (true_positive+false_positive)==0 else true_positive/(true_positive+false_positive)
            recall =0 if (true_positive + false_negative)==0 else true_positive/(true_positive+false_negative)

            sum_test_precision += precision
            sum_test_recall += recall

            print('test tp:%s tn:%s fp:%s fn:%s' % (true_positive, true_negative, false_positive, false_negative))
            # print('test predict:%s recall:%s' % (precision, recall))
            sigmoid_predict = np.squeeze(sigmoid_predict)
            precision_list.append(precision)
            recall_list.append(recall)
            logits_list.append(sigmoid_predict)
            num_list.append([true_positive, true_negative, false_positive, false_negative])

        sum_test_precision/=length_test
        sum_test_recall/=length_test
        sum_val_fscore = calFscore(sum_test_precision, sum_test_recall, 1)
        np.save('%s/model/predict.npy' % dataset_name, logits_list)
        np.save('%s/model/act.npy' % dataset_name, y_test)
        np.save('%s/model/precision_list.npy' % dataset_name, precision_list)
        np.save('%s/model/recall_list.npy' % dataset_name, recall_list)
        np.save('%s/model/num_list.npy' % dataset_name, num_list)

        print('sum validation precision:%s recall:%s fscore:%s ' % (sum_test_precision, sum_test_recall,sum_val_fscore))

def cross_matrix(labels, logits):
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)
    return -tf.reduce_mean(relu_logits - logits * labels+tf.log(tf.exp(neg_abs_logits)))


def corss_matrix_focal_loss(labels, logits,alpha,gamma):
    predict = tf.nn.sigmoid(logits)
    zeros=array_ops.zeros_like(predict,dtype=predict.dtype)

    pos_p=array_ops.where(labels>zeros,labels-predict,zeros)
    neg_p=array_ops.where(labels>zeros,zeros,predict)

    epsilon=1e-8
    return -tf.reduce_mean(alpha * (pos_p**gamma) *tf.log(tf.clip_by_value(predict, epsilon, 1.0)) +(1-alpha)*(neg_p**gamma)*tf.log(tf.clip_by_value(1-predict, epsilon, 1.0)))

def accuracy_matrix_new(labels, logits):
    labels=tf.cast(labels,dtype=tf.int32)
    logits=tf.cast(logits,dtype=tf.int32)

    true_positive = tf.reduce_sum(labels * logits)
    false_positive = tf.reduce_sum(logits) - true_positive
    false_negative = tf.reduce_sum(labels) - true_positive
    true_negative = tf.cast(tf.size(labels), dtype=tf.int32) - true_positive - false_positive - false_negative

    return true_positive, true_negative, false_positive, false_negative

def accuracy_matrix(labels, logits):
    # true_positive = tf.reduce_sum(labels * logits)
    # false_positive = tf.reduce_sum(logits) - true_positive
    # false_negative = tf.reduce_sum(labels) - true_positive
    # true_negative = tf.cast(tf.size(labels),dtype=tf.float32) - true_positive-false_positive-false_negative

    true_positive = tf.reduce_sum(tf.multiply(labels, logits))

    # false positive
    false_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labels, 0), tf.equal(logits, 1)), tf.float32))

    # false negative
    false_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labels, 1), tf.equal(logits, 0)), tf.float32))

    # true negative
    true_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(labels, 0), tf.equal(logits, 0)), tf.float32))

    return  true_positive,true_negative,false_positive,false_negative


def batchNorm(inputs, is_test, layer):
    x_shape = inputs.get_shape()
    params_shape = x_shape[-1:]

    bnepsilon = 1e-5
    bndecay = 0.999
    axis = list(range(len(x_shape) - 1))

    beta = tf.get_variable('beta%s' % layer, params_shape, initializer=tf.zeros_initializer, )
    gamma = tf.get_variable('gamma%s' % layer, params_shape, initializer=tf.ones_initializer)
    moving_mean = tf.get_variable('moving_mean%s' % layer, params_shape, initializer=tf.zeros_initializer,
                                  trainable=False)
    moving_variance = tf.get_variable('moving_variance%s' % layer, params_shape, initializer=tf.ones_initializer,
                                      trainable=False)

    mean, variance = tf.nn.moments(inputs, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, bndecay)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, bndecay)

    m, v = tf.cond(is_test, lambda: (moving_mean, moving_variance), lambda: (mean, variance))
    ybn = tf.nn.batch_normalization(inputs, m, v, beta, gamma, bnepsilon)
    return ybn

def calTopN(label,predict):
    predict = np.squeeze(predict, axis=(0, 3))
    N = int(np.sum(label))
    predict_list = predict.flatten()
    top_N = heapq.nlargest(int(N), predict_list)
    previous = top_N[0]
    x, y = np.where(predict == previous)
    xs = x
    ys = y
    for j in range(1, len(top_N)):
        if top_N[j] == previous:
            continue
        x, y = np.where(predict == top_N[j])
        xs = np.concatenate((xs, x))
        ys = np.concatenate((ys, y))
        previous = top_N[j]
    act_matrix = label[(xs, ys)]

    tp = np.sum(act_matrix)
    precision = tp / np.size(act_matrix)
    return precision


def calFscore(precision,recall,alpha):
    if (precision==0)|(recall==0):
        F_score=0
    else:
        F_score=((alpha*alpha+1)*precision*recall)/(alpha*alpha*precision+recall)
    return F_score
def calMCC(tp,tn,fp,fn):
    return (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

def tempData():
    x=[]
    y=[]
    x.append(np.random.randint(-10,10,[10,20,80]).astype(np.float32))
    x.append(np.random.randint(-10,10,[20,30,80]).astype(np.float32))
    y.append(np.random.randint(0,1,[10,20,2]).astype(np.int32))
    y.append(np.random.randint(0,1,[20,30,2]).astype(np.int32))
    return x,y,x,y

def readLabel(type,dataset_name):
    label = []
    path = '%s/kfold/%s' % (dataset_name, 10)
    lists = os.listdir('%s/data/%s/input' % (path, type))

    for i in range(0, len(lists)):
        label = np.concatenate((label, np.load('%s/data/%s/label/label%s.npy' % (path, type, i))), axis=0)
    return label

def readSingleData(dataset_name,type):
    x = np.load('%s/newdata/%s/x/x_%d.npy' % (dataset_name, type, 2))
    y = np.load('%s/newdata/%s/y/y_%d.npy' % (dataset_name, type, 2))
    print(np.shape(x)[0])
    return x, y
def shuffle_data(x,y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_dataset = x[permutation]
    shuffled_labels = y[permutation]
    return shuffled_dataset,shuffled_labels

def GroupNorm(x,  G, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N,  H, W ,C= x.shape

    gamma = tf.get_variable('gamma', [C],
                            initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable('beta', [C],
                           initializer=tf.constant_initializer(0.0))
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N,  H, W,C])
    return x * gamma + beta

def countNumber(y):
    sum_size=0
    for i in y:
        sum_size+=np.size(i)
    print(sum_size)
if __name__ == '__main__':
    dataset_name = 'dataset3'

    if not os.path.exists('%s/out'%dataset_name):
        os.makedirs('%s/out/train'%dataset_name)
        os.makedirs('%s/out/test'%dataset_name)
        os.makedirs('%s/model'%dataset_name)
        os.makedirs('%s/traingroup'%dataset_name)

    x,y = readData(dataset_name)
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)
    x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=0.5)
    fcnModelTf(x_train, x_val, x_test, y_train, y_val, y_test)
