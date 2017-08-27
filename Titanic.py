import tensorflow as tf
import pandas as pd
import numpy as np
import bmTools
# tData = pd.read_csv(r'D:\datas\train.csv')
# tData['Sex'] = tData['Sex'].apply(lambda s:1 if s == 'male' else 0)
# tData = tData.fillna(0)
# dataSetX = tData[['Sex','Age','Pclass','SibSp','Parch','Fare']]
# dataSetX = dataSetX.as_matrix()
# tData['Deceased'] = tData['Survived'].apply(lambda s: int(not s))
# dataSetY = tData[['Survived','Deceased']]
# dataSetY = dataSetY.as_matrix()
# x_test, y_test, x_train, y_train = bmTools.splitDataSet(dataSetX,dataSetY)
x = tf.placeholder(tf.float32, shape=[None,6])
y = tf.placeholder(tf.float32, shape=[None,2])
W = tf.Variable(tf.random_normal([6,2]), name = 'weights')
b = tf.Variable(tf.zeros([2]), name= 'bias')
saver = tf.train.Saver()
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
# cross_entropy = -tf.reduce_sum(y * tf.log(y_pred + 1e-10),reduction_indices=1)
# cost = tf.reduce_mean(cross_entropy)
# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     tf.global_variables_initializer().run()
#     for epoch in range(30):
#         total_loss = 0
#         for i in range(len(x_train)):
#             feed = {x:[x_train[i]],y:[y_train[i]]}
#             _,loss = sess.run([train_op, cost], feed_dict=feed)
#             total_loss += loss
#         print('Epoch: %04d,total loss=%.9f' %(epoch + 1,total_loss))
#     print('Training complete')
#     save_path = saver.save(sess, 'D:\datas\model.ckpt')
#     pred = sess.run(y_pred,feed_dict={x:x_test})
#     correct = np.equal(np.argmax(pred,1),np.argmax(y_test,1))
#     accuracy = np.mean(correct.astype(np.float32))
#     print('accuracy is %.9f'% accuracy)



testData = pd.read_csv(r'D:\datas\test.csv')
testData = testData.fillna(0)
testData['Sex'] = testData['Sex'].apply(lambda s: 1 if s == 'male' else 0)
# saver = tf.train.Saver()
x_test = testData[['Sex','Age','Pclass','SibSp','Parch','Fare']]
with tf.Session() as sess:
    saver.restore(sess,r'D:\datas\model.ckpt')
    pred = np.argmax(sess.run(y_pred,feed_dict={x:x_test}))
    print(pred)