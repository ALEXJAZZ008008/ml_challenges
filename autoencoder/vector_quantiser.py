# Copyright University College London 2023, 2024
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# For internal research only.


import einops
import tensorflow as tf


# https://keras.io/examples/generative/vq_vae/
class VectorQuantiser(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        initialiser = tf.keras.initializers.Zeros()
        self.mean = tf.Variable(initial_value=initialiser(shape=(1,)), trainable=True)

        initialiser = tf.keras.initializers.Ones()
        self.stddev = tf.Variable(initial_value=initialiser(shape=(1,)), trainable=True)

        # Initialize the embeddings which we will quantize.
        initialiser = tf.keras.initializers.RandomNormal(stddev=1.0)
        self.embeddings = tf.Variable(initial_value=initialiser(shape=(self.embedding_dim, self.num_embeddings)),
                                      trainable=True)

    def get_discretised(self, x):
        # Calculate L2-normalized distance between the inputs and the embedding.
        # Derive the indices for minimum distances.
        discretised = tf.math.argmin(((tf.math.reduce_sum(tf.math.pow(x, 2.0), axis=1, keepdims=True) +
                                       tf.math.reduce_sum(tf.math.pow(self.embeddings, 2.0), axis=0)) -
                                      (2.0 * tf.matmul(x, self.embeddings))), axis=1)

        return discretised

    def call(self, x, **kwargs):
        standardised = tf.math.divide_no_nan(x, self.stddev) - self.mean

        # You can learn more about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/.
        self.add_loss(tf.math.reduce_mean(tf.math.pow(tf.math.reduce_mean(x, axis=[1, 2, 3]) - self.mean, 2.0)))
        self.add_loss(tf.math.reduce_mean(tf.math.pow(tf.math.reduce_std(x, axis=[1, 2, 3]) - self.stddev, 2.0)))

        standardised = einops.rearrange(standardised, "b h w c -> b c (h w)")

        # Calculate the input shape of the inputs and then flatten the inputs keeping `embedding_dim` intact.
        discretised = self.get_discretised(tf.reshape(standardised, [-1, self.embedding_dim]))
        # Quantization.
        # Reshape the quantized values back to the original input shape
        quantised = tf.reshape(tf.matmul(tf.one_hot(discretised, self.num_embeddings), tf.transpose(self.embeddings)),
                               tf.shape(standardised))

        # Calculate vector quantization loss and add that to the layer.
        # You can learn more about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/.
        # Check the original paper to get a handle on the formulation of the loss function.
        self.add_loss(tf.math.reduce_mean(tf.math.pow(tf.stop_gradient(standardised) - quantised, 2.0)))

        quantised = einops.rearrange(quantised, "b c (h w) -> b h w c", w=x.shape[2])
        discretised = tf.reshape(discretised, [-1, quantised.shape[-1]])

        quantised = (quantised + self.mean) * self.stddev

        # Straight-through estimator.
        quantised = x + tf.stop_gradient(quantised - x)

        return quantised, discretised
