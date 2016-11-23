import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # test data set of mini digest

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

nodes_number_hl1 = 1000
nodes_number_hl2 = 1000
nodes_number_hl3 = 1000

classes_num = 10  # {0,1...,9}
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layer1 = {'weights': tf.Variable(tf.random_normal([784, nodes_number_hl1])),
                     'biases': tf.Variable(tf.random_normal(nodes_number_hl1))}

    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([nodes_number_hl1, nodes_number_hl2])),
                     'biases': tf.Variable(tf.random_normal(nodes_number_hl2))}

    hidden_layer3 = {'weights': tf.Variable(tf.random_normal([nodes_number_hl2, nodes_number_hl3])),
                     'biases': tf.Variable(tf.random_normal(nodes_number_hl3))}

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_number_hl3, classes_num])),
                     'biases': tf.Variable(tf.random_normal([classes_num]))}

    # (input_data * weights ) + biases
    layer1 = tf.add(tf.matmul(data, hidden_layer1['weights']) + hidden_layer1['biases'])
    layer1 = tf.nn.relu(layer1) # activation function

    layer2 = tf.add(tf.matmul(layer1, hidden_layer2['weights']) + hidden_layer2['biases'])
    layer2 = tf.nn.relu(layer2)  # activation function

    layer3 = tf.add(tf.matmul(layer2, hidden_layer3['weights']) + hidden_layer3['biases'])
    layer3 = tf.nn.relu(layer3)  # activation function

    output = tf.matmul(layer3, output_layer['weights']) + output_layer['biases']
    return output
