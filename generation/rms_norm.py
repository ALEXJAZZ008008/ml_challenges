# Copyright University College London 2023, 2024
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# For internal research only.


import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        initialiser = tf.keras.initializers.Ones()
        self.scale = tf.Variable(initial_value=initialiser(shape=(1,)), trainable=True)

    def call(self, x, **kwargs):
        normalised = (tf.norm(x, axis=-1) * self.scale) * tf.math.pow(tf.cast(x.shape[1], dtype=tf.float32), 0.5)

        return normalised
