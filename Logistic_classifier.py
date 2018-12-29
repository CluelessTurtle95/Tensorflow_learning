
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris  = load_iris()

iris_X = iris['data'][ 0:-1 , :]
iris_y = iris['target'][:-1]
iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

numFeatures = trainX.shape[1]

numLabels = trainY.shape[1]

X = tf.placeholder(tf.float32 , [None , numFeatures] )
yGold = tf.placeholder(tf.float32 , [None , numLabels])

W = tf.Variable(tf.random_uniform( [4 , 3] ))
b = tf.Variable(tf.zeros([3]))

y = tf.nn.sigmoid(tf.matmul(X , W) + b)
loss = tf.reduce_mean( -1 * (yGold* tf.log(y) + (1-yGold)*tf.log(1 - y)))
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

loss_values = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(500):
        g , loss_val = sess.run([train , loss] , feed_dict = {X : trainX , yGold : trainY} )
        loss_values.append(loss_val)
    wf = sess.run(W)
    bf = sess.run(b)


predict = np.equal(np.argmax(1 / (1 + np.exp( -1 * (testX @ wf + bf))) , axis = 1) , np.argmax( testY , axis = 1))

final = np.mean(predict)

print("Model Accuracy : {} Percent".format(final*100))

plt.plot(loss_values)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()