import tensorflow as tf
import numpy as np

sess = tf.Session()

a = tf.placeholder(dtype=tf.float16, shape=[2, 2],name='a')
b = tf.placeholder(dtype=tf.float16, shape=[2, 2],name='b')

add = tf.add(a, b)
sess = tf.Session()
ret = sess.run(add, feed_dict={a: np.random.randint(0, 10, size=(2, 2)), b: np.random.randint(0, 10, size=(2, 2))})
print(ret)
save_path = 'saver/fns.ckpt'

#Create a saver

'''python v1 = tf.Variable(..., name='v1') v2 = tf.Variable(..., name='v2')
# Pass the variables as a dict: saver = tf.train.Saver({'v1': v1, 'v2': v2})
# Or pass them as a list. saver = tf.train.Saver([v1, v2]) 
#  Passing a list is equivalent to passing a dict with the variable op names 
#  as keys: saver = tf.train.Saver({v.op.name: v for v in [v1, v2]}) '''
v1 = tf.Variable(0, name='v1')
v2 = tf.Variable(0, name='v2')
saver=tf.train.Saver(var_list=[v1])
#Launch the graph and train, saving the model every 1,000 steps.

res = saver.save(sess, save_path)
