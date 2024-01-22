# Copyright University College London 2023, 2024
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# For internal research only.


import tensorflow as tf


class Standardiser(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        initialiser = tf.keras.initializers.Zeros()
        self.mean = tf.Variable(initial_value=initialiser(shape=(1,)), trainable=True)

        initialiser = tf.keras.initializers.Ones()
        self.stddev = tf.Variable(initial_value=initialiser(shape=(1,)), trainable=True)

    def call(self, x, **kwargs):
        standardised = tf.math.divide_no_nan(x, self.stddev) - self.mean

        self.add_loss(tf.math.reduce_mean(tf.math.pow(tf.math.reduce_mean(x, axis=[1, 2, 3]) - self.mean, 2.0)))
        self.add_loss(tf.math.reduce_mean(tf.math.pow(tf.math.reduce_std(x, axis=[1, 2, 3]) - self.stddev, 2.0)))

        return standardised
