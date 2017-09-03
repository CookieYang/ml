import tensorflow as tf
def maxPoolLayer(x, kHeight,kWidth,strideX,strideY,name,padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,kHeight,kWidth,1],strides=[1,strideX,strideY,1],padding=padding,
                          name = name)
def dropout(x, keepPro, name=None):
    return tf.nn.dropout(x,keepPro,name)

def LRN(x,R,alpha,beta,name = None,bias= 1.0):
    return tf.nn.local_response_normalization(x,depth_radius = R, alpha = alpha, beta = beta,
                                              bias = bias, name = name)