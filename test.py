
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告
def he():
    global X
    X = 1024

def wo():
    global X
    print(X)
import tensorflow as tf
if __name__ == '__main__':
    he()
    wo()


    hello = tf.constant('Hello ,TensorFlow',dtype=object)
    sess = tf.Session()
    print(sess.run(hello).decode('utf-8'))