import tensorflow as tf


def resConv2d(x, W):
    conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')
    return conv
def conv2d(x, W, bias,name):
    conv=tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')
    return  tf.nn.bias_add(conv,bias,name=name)
def atrous_conv2d(x, W, bias,rate=2):
    conv = tf.nn.atrous_conv2d(x, W, rate, padding='SAME')
    return tf.nn.bias_add(conv, bias)
# def conv2d_transpose(x, b, output_filters, stride = 2):
#     conv = tf.layers.conv2d_transpose(inputs=x,filters=output_filters,kernel_size=2,strides=(stride,stride),padding='same')
#     return tf.nn.bias_add(conv, b)

def conv2d_transpose(x, W, b, output_shape=None, stride = 2,name=None):
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b,name=name)

def max_pool(x,strides=2,name='None'):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,strides,strides,1],padding='SAME',name=name)

def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, stddev=0.02, name=None):
    with tf.name_scope('weight'):
        # initial = tf.truncated_normal(shape, stddev=stddev,seed=1)
        initial = tf.glorot_uniform_initializer(seed=1)
        if name is None:
            return tf.Variable(initial)
        else:
            weight=tf.get_variable(name, shape=shape,initializer=initial)
            # tf.add_to_collection('ws', weight)
            tf.summary.histogram(name, weight)
            return weight

    # W=tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    # return W



def bias_variable(shape, name=None):
    with tf.name_scope('bias'):
        initial = tf.constant(0.0, shape=shape)
        if name is None:
            return tf.Variable(initial)
        else:
            bias=tf.get_variable(name, initializer=initial)
            # tf.add_to_collection('bs', bias)
            tf.summary.histogram(name, bias)
            return bias

def conv1d(x,W):
    conv=tf.nn.conv1d(x,W,[1,1,1],padding='SAME')
    # return tf.nn.bias_add(conv,bias)
    return conv

def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))

