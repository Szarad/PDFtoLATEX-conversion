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

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y )) # cost function
    optimizer = tf.train.AdamOptimizer().minimize(cost) #learning_rate =0.001

    # cycles feed forward + backprop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        #########training

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        ###########training

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)



