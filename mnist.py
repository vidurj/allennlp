from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import heapq
import random
import sys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class Model:
    def __init__(self, training_size):
        batch_size = None
        self.x = tf.placeholder(tf.float32, shape=[batch_size, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[batch_size, 10])
        self.indices = tf.placeholder(tf.int32, shape=[batch_size])
        self.lmbd = tf.placeholder(tf.float32, shape=[1])
        weights = tf.Variable(tf.constant(0.99, shape=[training_size, 1]))
        global_weights = tf.nn.sigmoid(weights)

        print('shape 0', global_weights)

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.relevant_weights = tf.squeeze(tf.nn.embedding_lookup(global_weights, self.indices))
        print('shape 1', self.relevant_weights)
        self.relevant_weights_total = tf.reduce_mean(self.relevant_weights)
        # tf.reduce_mean(
        print(self.y_, y_conv)
        temp = self.relevant_weights * tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv)
        print('shape 2', temp)
        self.cross_entropy = tf.reduce_sum(temp) / tf.reduce_sum(self.relevant_weights)
        self.logits = tf.nn.softmax(y_conv)
        self.total_weight = tf.reduce_sum(global_weights)
        self.regularization = tf.abs(self.lmbd - self.total_weight) / 1000.0
        loss = self.cross_entropy + self.regularization
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.train_step_regularized = tf.train.AdamOptimizer(1e-4).minimize(loss)
        self.only_train_weights = tf.train.AdamOptimizer(0.01).minimize(loss, var_list=[weights])
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_eval_preds(xes, yes, model, legal_indices, batch_size=1024):
    choices = []
    for i in range(0, len(xes), batch_size):
        preds = \
            model.logits.eval(
                feed_dict={model.x: xes[i: i + batch_size], model.y_: yes[i: i + batch_size],
                           model.keep_prob: 1.0})
        entropies = - np.sum(preds * np.log(preds), axis=1)
        for offset, x in enumerate(entropies):
            index = offset + i
            if index in legal_indices:
                heapq.heappush(choices, (x, index))
            if len(choices) > 10:
                heapq.heappop(choices)
    chosen_indices = []
    for _, index in choices:
        legal_indices.remove(index)
        chosen_indices.append(index)
    return chosen_indices


def noise_data(mnist_data):
    xes = mnist_data.images
    zero_label = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    noised_indices = {i for i in range(len(mnist_data.labels)) if random.random() < 0.2}
    yes = [y if i not in noised_indices else zero_label for i, y in enumerate(mnist_data.labels)]
    return xes, yes, noised_indices


def main():
    random.seed(a=0)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    print(len(mnist.train.images), len(mnist.validation.images), len(mnist.test.images))
    print('-' * 100)
    train_xes, train_yes, train_noised_indices = noise_data(mnist.train)

    # print(train_noised_indices)
    train_data = list(enumerate(zip(train_xes, train_yes)))
    corrupted_data = [(i, point) for i, point in train_data if i in train_noised_indices]
    correct_data = [(i, point) for i, point in train_data if i not in train_noised_indices]
    if sys.argv[1] == 'skip_noised':
        train_data = [(i, (x, y)) for (i, (x, y)) in train_data if i not in train_noised_indices]
    else:
        assert sys.argv[1] == 'keep_noised'

    model = Model(len(mnist.train.images))
    all_train_indices = []
    batch_size = 1024

    test_xes, test_yes, test_noised_indices = noise_data(mnist.validation)
    lmbd = len(train_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10000):
            cur_index = 0
            if epoch > 30:
                lmbd = len(correct_data)
            random.shuffle(train_data)
            while cur_index + batch_size < len(train_data):
                _, _, total_weight, reg_loss = sess.run([model.train_step, model.only_train_weights, model.total_weight, model.regularization],
                         feed_dict={
                             model.x: [x for (i, (x, y)) in
                                       train_data[cur_index: cur_index + batch_size]],
                             model.y_: [y for (i, (x, y)) in
                                        train_data[cur_index: cur_index + batch_size]],
                             model.keep_prob: 0.5,
                             model.lmbd: [lmbd],
                             model.indices: [i for (i, (x, y)) in
                                        train_data[cur_index: cur_index + batch_size]]
                         })
                cur_index += batch_size
            print('epoch', epoch, 'lmd', lmbd, 'total weight', total_weight, 'regularization', reg_loss)

            test_size = 1024
            loss_on_correct, weight_on_correct = sess.run([model.cross_entropy, model.relevant_weights_total],
                feed_dict={
                    model.x: [x for (i, (x, y)) in correct_data[0: test_size]],
                    model.y_: [y for (i, (x, y)) in correct_data[0: test_size]],
                    model.keep_prob: 0.5,
                    model.lmbd: [lmbd],
                    model.indices: [i for (i, (x, y)) in correct_data[0: test_size]]
                })

            loss_on_corrupted, weight_on_corrupted = sess.run([model.cross_entropy, model.relevant_weights_total],
                feed_dict={
                    model.x: [x for (i, (x, y)) in corrupted_data[0: test_size]],
                    model.y_: [y for (i, (x, y)) in corrupted_data[0: test_size]],
                    model.keep_prob: 0.5,
                    model.lmbd: [lmbd],
                    model.indices: [i for (i, (x, y)) in corrupted_data[0: test_size]]
                })

            print('Loss on correct {} loss on corrupted {}'.format(loss_on_correct,
                                                                   loss_on_corrupted))
            print('Weight on correct {} weight on corrupted {}'.format(weight_on_correct,
                                                                   weight_on_corrupted))

            print('test accuracy %g' % model.accuracy.eval(
                feed_dict={
                    model.x: test_xes,
                    model.y_: test_yes,
                    model.keep_prob: 1.0
                }))


if __name__ == '__main__':
    main()
