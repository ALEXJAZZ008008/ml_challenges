# Copyright University College London 2023, 2024
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# For internal research only.


import einops
import tensorflow as tf


commitment_loss_weight = 0.25


# https://keras.io/examples/generative/vq_vae/
class VectorQuantiser(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Initialize the embeddings which we will quantize.
        initialiser = tf.keras.initializers.Orthogonal()
        self.embeddings = tf.Variable(initial_value=initialiser(shape=(self.embedding_dim, self.num_embeddings)),
                                      trainable=True)

    def get_discretised(self, continuous):
        # Calculate L2-normalized distance between the inputs and the embedding.
        # Derive the indices for minimum distances.
        discretised = tf.math.argmin(((tf.math.reduce_sum(tf.math.pow(continuous, 2.0), axis=1, keepdims=True) +
                                       tf.math.reduce_sum(tf.math.pow(self.embeddings, 2.0), axis=0)) -
                                      (2.0 * tf.matmul(continuous, self.embeddings))), axis=1)

        return discretised

    def call(self, x, **kwargs):
        continuous = einops.rearrange(x, "b h w c -> b c (h w)")

        # Calculate the input shape of the inputs and then flatten the inputs keeping `embedding_dim` intact.
        discretised = self.get_discretised(tf.reshape(continuous, [-1, self.embedding_dim]))
        # Quantization.
        # Reshape the quantized values back to the original input shape
        quantised = tf.reshape(tf.matmul(tf.one_hot(discretised, self.num_embeddings), tf.transpose(self.embeddings)),
                               tf.shape(continuous))

        # Calculate vector quantization loss and add that to the layer.
        # You can learn more about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/.
        # Check the original paper to get a handle on the formulation of the loss function.
        self.add_loss(tf.math.reduce_mean(tf.math.pow(tf.stop_gradient(continuous) - quantised, 2.0)))
        self.add_loss(commitment_loss_weight *
                      tf.math.reduce_mean(tf.math.pow(continuous - tf.stop_gradient(quantised), 2.0)))

        quantised = einops.rearrange(quantised, "b c (h w) -> b h w c", h=x.shape[1], w=x.shape[2])
        discretised = tf.reshape(discretised, [-1, quantised.shape[-1]])

        # Straight-through estimator.
        quantised = x + tf.stop_gradient(quantised - x)

        return quantised, discretised
