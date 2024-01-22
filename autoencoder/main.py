# Copyright University College London 2023, 2024
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# For internal research only.


import os
import shutil
import errno
import random
import copy
import numpy as np
import scipy
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.filters import gaussian, unsharp_mask
from sklearn.preprocessing import StandardScaler
import einops
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm
from standardiser import Standardiser
from vector_quantiser import VectorQuantiser
import pickle
from PIL import Image


np.seterr(all="print")


dataset_name = "cifar10"
output_path = "../output/autoencoder/"

read_data_from_storage_bool = False

preprocess_list_bool = False
greyscale_bool = False
min_output_dimension_size = 32
max_output_dimension_size = 32

deep_bool = True
conv_bool = True
alex_bool = True
discrete_bool = True
gaussian_negative_log_likelihood_bool = False
dense_layers = 4
conditional_dense_layers = 2
filters = [64, 128, 256, 512, 1024]
conv_layers = [2, 2, 2, 2, 2]
latent_shape = (2, 2, filters[-1])
num_heads = 4
key_dim = 32
num_embeddings_multiplier = 64.0

if alex_bool:
    learning_rate = 1e-04
    weight_decay = 0.0
    ema_overwrite_frequency = None
else:
    learning_rate = 1e-04
    weight_decay = 0.0
    ema_overwrite_frequency = None

gradient_accumulation_bool = False

epochs = 256

if gradient_accumulation_bool:
    min_batch_size = 32
    max_batch_size = 32
else:
    if alex_bool:
        min_batch_size = 16
        max_batch_size = 16
    else:
        min_batch_size = 32
        max_batch_size = 32

axis_zero_flip_bool = False
axis_one_flip_bool = False
gaussian_bool = False
max_sigma = 1.0
sharpen_bool = False
max_radius = 1.0
max_amount = 1.0
scale_bool = False
min_scale = 0.75
max_scale = 1.25
rotate_bool = False
min_angle = -45.0
max_angle = 45.0
translate_bool = False
translate_proportion = 0.25

if alex_bool:
    mean_squared_error_epoch = 1
else:
    mean_squared_error_epoch = epochs

gaussian_latent_loss_weight = 0.0

test_batch_size = min_batch_size


def mkdir_p(path):
    try:
        os.makedirs(path, mode=0o770)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

    return True


def get_input():
    print("get_input")

    x_train_images_output_path = "{0}/x_train_images/".format(output_path)
    x_test_images_output_path = "{0}/x_test_images/".format(output_path)

    if read_data_from_storage_bool:
        mkdir_p(x_train_images_output_path)
        mkdir_p(x_test_images_output_path)

    dataset = tfds.load(dataset_name)

    dataset_train = dataset["train"]
    dataset_test = dataset["test"]

    x_train_images = []
    x_test_images = []

    x_train_labels = []
    x_test_labels = []

    for i, example in enumerate(dataset_train):
        if read_data_from_storage_bool:
            current_x_train_images = example["image"].numpy().astype(np.float32)
            x_train_labels.append(example["label"].numpy().astype(np.float32))

            x_train_images.append("{0}/{1}.pkl".format(x_train_images_output_path, str(i)))

            with open(x_train_images[-1], "wb") as file:
                pickle.dump(current_x_train_images, file)
        else:
            x_train_images.append(example["image"].numpy().astype(np.float32))
            x_train_labels.append(example["label"].numpy().astype(np.float32))

    for i, example in enumerate(dataset_test):
        if read_data_from_storage_bool:
            current_x_test_images = example["image"].numpy().astype(np.float32)
            x_test_labels.append(example["label"].numpy().astype(np.float32))

            x_test_images.append("{0}/{1}.pkl".format(x_test_images_output_path, str(i)))

            with open(x_test_images[-1], "wb") as file:
                pickle.dump(current_x_test_images, file)
        else:
            x_test_images.append(example["image"].numpy().astype(np.float32))
            x_test_labels.append(example["label"].numpy().astype(np.float32))

    x_train_labels = np.array(x_train_labels)
    x_test_labels = np.array(x_test_labels)

    x_train_labels = np.expand_dims(x_train_labels, axis=-1)
    x_test_labels = np.expand_dims(x_test_labels, axis=-1)

    return x_train_images, x_test_images, x_train_labels, x_test_labels


def get_data_from_storage(data):
    if read_data_from_storage_bool:
        with open(data, "rb") as file:
            data = pickle.load(file)

    return data


def set_data_from_storage(data, current_output_path):
    if read_data_from_storage_bool:
        with open(current_output_path, "wb") as file:
            pickle.dump(data, file)

        data = current_output_path

    return data


def convert_rgb_to_greyscale(images):
    print("convert_rgb_to_greyscale")

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        if image.shape[-1] > 1:
            image = rgb2gray(image)
            image = np.expand_dims(image, axis=-1)

        images[i] = set_data_from_storage(image, images[i])

    return images


def get_next_geometric_value(an, a0):
    n = np.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2.0, (np.ceil(n) - 1.0))

    return an


def rescale_images_list(images):
    print("rescale_images")

    max_dimension_size = -1

    images_len = len(images)

    for i in range(images_len):
        image = get_data_from_storage(images[i])

        current_max_dimension_size = np.max(image.shape[:-1])

        if current_max_dimension_size > max_dimension_size:
            max_dimension_size = current_max_dimension_size

    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    if min_output_dimension_size is not None:
        if output_dimension_size < min_output_dimension_size:
            output_dimension_size = min_output_dimension_size

    if max_output_dimension_size is not None:
        if output_dimension_size > max_output_dimension_size:
            output_dimension_size = max_output_dimension_size

    for i in range(images_len):
        image = get_data_from_storage(images[i])

        image = rescale(image, output_dimension_size / np.max(image.shape[:-1]), order=3, preserve_range=True,
                        channel_axis=-1)

        images[i] = set_data_from_storage(image, images[i])

    return images


def normalise_images_list(x_train_images, x_test_images):
    print("normalise_images")

    standard_scaler = StandardScaler()

    x_train_images_len = len(x_train_images)

    for i in range(x_train_images_len):
        current_x_train_images = get_data_from_storage(x_train_images[i])

        standard_scaler.partial_fit(np.reshape(current_x_train_images, (-1, 1)))

    for i in range(x_train_images_len):
        current_x_train_images = get_data_from_storage(x_train_images[i])

        current_x_train_images = np.reshape(standard_scaler.transform(np.reshape(current_x_train_images,
                                                                                 (-1, 1))),
                                            current_x_train_images.shape)

        x_train_images[i] = set_data_from_storage(current_x_train_images, x_train_images[i])

    for i in range(len(x_test_images)):
        current_x_test_images = get_data_from_storage(x_test_images[i])

        current_x_test_images = np.reshape(standard_scaler.transform(np.reshape(current_x_test_images,
                                                                                (-1, 1))),
                                           current_x_test_images.shape)

        x_test_images[i] = set_data_from_storage(current_x_test_images, x_test_images[i])

    return x_train_images, x_test_images, standard_scaler


def pad_image(image, output_dimension_size):
    while image.shape[0] + 1 < output_dimension_size:
        image = np.pad(image, ((1, 1), (0, 0), (0, 0)))  # noqa

    if image.shape[0] < output_dimension_size:
        image = np.pad(image, ((0, 1), (0, 0), (0, 0)))  # noqa

    while image.shape[1] + 1 < output_dimension_size:
        image = np.pad(image, ((0, 0), (1, 1), (0, 0)))  # noqa

    if image.shape[1] < output_dimension_size:
        image = np.pad(image, ((0, 0), (0, 1), (0, 0)))  # noqa

    return image


def pad_images_list(images):
    print("pad_images")

    max_dimension_size = -1

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        current_max_dimension_size = np.max(image.shape[:-1])

        if current_max_dimension_size > max_dimension_size:
            max_dimension_size = current_max_dimension_size

    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    padding_masks = copy.deepcopy(images)

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        padding_mask = np.ones(image.shape, dtype=np.float32)

        image = pad_image(image, output_dimension_size)
        padding_mask = pad_image(padding_mask, output_dimension_size)

        images[i] = set_data_from_storage(image, images[i])

        if read_data_from_storage_bool:
            current_padding_masks_split = padding_masks[i].strip().split('.')
            padding_masks[i] = "{0}_padding_mask.{1}".format(current_padding_masks_split[0],
                                                             current_padding_masks_split[1])

        padding_masks[i] = set_data_from_storage(padding_mask, padding_masks[i])

    return images, padding_masks


def convert_images_to_tensor_list(images):
    print("convert_images_to_tensor")

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        image = tf.convert_to_tensor(image)

        images[i] = set_data_from_storage(image, images[i])

    return images


def preprocess_images_list(x_train_images, x_test_images):
    print("preprocess_images")

    x_train_images = rescale_images_list(x_train_images)
    x_test_images = rescale_images_list(x_test_images)

    x_train_images, x_test_images, standard_scaler = normalise_images_list(x_train_images, x_test_images)

    x_train_images, x_train_padding_masks = pad_images_list(x_train_images)
    x_test_images, x_test_padding_masks = pad_images_list(x_test_images)

    x_train_images = convert_images_to_tensor_list(x_train_images)
    x_test_images = convert_images_to_tensor_list(x_test_images)

    x_train_padding_masks = convert_images_to_tensor_list(x_train_padding_masks)

    return x_train_images, x_train_padding_masks, x_test_images, standard_scaler


def rescale_images_array(images):
    print("rescale_images")

    max_dimension_size = np.max(images.shape[1:-1])
    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    if min_output_dimension_size is not None:
        if output_dimension_size < min_output_dimension_size:
            output_dimension_size = min_output_dimension_size

    if max_output_dimension_size is not None:
        if output_dimension_size > max_output_dimension_size:  # noqa
            output_dimension_size = max_output_dimension_size

    rescaled_images = []

    for i in range(len(images)):
        rescaled_images.append(rescale(images[i], output_dimension_size / max_dimension_size, order=3,
                                       preserve_range=True, channel_axis=-1))

    images = np.array(rescaled_images)

    return images


def normalise_images_array(x_train_images, x_test_images):
    print("normalise_images")

    standard_scaler = StandardScaler()

    x_train_images = np.reshape(standard_scaler.fit_transform(np.reshape(x_train_images, (-1, 1))),
                                x_train_images.shape)
    x_test_images = np.reshape(standard_scaler.transform(np.reshape(x_test_images, (-1, 1))),
                               x_test_images.shape)

    return x_train_images, x_test_images, standard_scaler


def pad_images_array(images):
    print("pad_images")

    max_dimension_size = np.max(images.shape[1:-1])
    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    while images.shape[1] + 1 < output_dimension_size:
        images = np.pad(images, ((0, 0), (1, 1), (0, 0), (0, 0)))  # noqa

    if images.shape[1] < output_dimension_size:
        images = np.pad(images, ((0, 0), (0, 1), (0, 0), (0, 0)))  # noqa

    while images.shape[2] + 1 < output_dimension_size:
        images = np.pad(images, ((0, 0,), (0, 0), (1, 1), (0, 0)))  # noqa

    if images.shape[2] < output_dimension_size:
        images = np.pad(images, ((0, 0), (0, 0), (0, 1), (0, 0)))  # noqa

    return images


def preprocess_images_array(x_train_images, x_test_images):
    print("preprocess_images")

    x_train_images = np.array(x_train_images)
    x_test_images = np.array(x_test_images)

    x_train_images = rescale_images_array(x_train_images)
    x_test_images = rescale_images_array(x_test_images)

    x_train_images, x_test_images, standard_scaler = normalise_images_array(x_train_images, x_test_images)

    x_train_padding_masks = np.ones(x_train_images.shape, dtype=np.float32)

    x_train_images = pad_images_array(x_train_images)
    x_test_images = pad_images_array(x_test_images)

    x_train_padding_masks = pad_images_array(x_train_padding_masks)

    x_train_images = tf.convert_to_tensor(x_train_images)
    x_test_images = tf.convert_to_tensor(x_test_images)

    x_train_padding_masks = tf.convert_to_tensor(x_train_padding_masks)

    return x_train_images, x_train_padding_masks, x_test_images, standard_scaler


def preprocess_labels(x_train_labels, x_test_labels):
    print("preprocess_labels")

    x_train_labels = tf.convert_to_tensor(x_train_labels)
    x_test_labels = tf.convert_to_tensor(x_test_labels)

    return x_train_labels, x_test_labels


def preprocess_input(x_train_images, x_test_images, x_train_labels, x_test_labels):
    print("preprocess_input")

    if greyscale_bool:
        x_train_images = convert_rgb_to_greyscale(x_train_images)
        x_test_images = convert_rgb_to_greyscale(x_test_images)

    if read_data_from_storage_bool or preprocess_list_bool:
        x_train_images, x_train_padding_masks, x_test_images, standard_scaler = preprocess_images_list(x_train_images,
                                                                                                       x_test_images)
    else:
        x_train_images, x_train_padding_masks, x_test_images, standard_scaler = preprocess_images_array(x_train_images,
                                                                                                        x_test_images)

    x_train_labels, x_test_labels = preprocess_labels(x_train_labels, x_test_labels)

    return x_train_images, x_train_padding_masks, x_test_images, standard_scaler, x_train_labels, x_test_labels


def get_previous_geometric_value(an, a0):
    n = np.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2.0, (np.floor(n) - 1.0))

    return an


def get_model_dense(x_train_images):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        input_shape = current_x_train_images.shape
    else:
        input_shape = x_train_images.shape[1:]

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    x = tf.keras.layers.Flatten()(x)

    x_flatten = x

    units = get_next_geometric_value(x.shape[-1], 2.0)

    for i in range(dense_layers - 1):
        units = get_previous_geometric_value(units - 1, 2.0)

    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=x_flatten.shape[-1])(x)
    x = tf.keras.layers.Reshape(x_input.shape[1:])(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_model_deep_dense(x_train_images):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train_images.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    x = tf.keras.layers.Flatten()(x)

    x_flatten = x

    x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1], 2.0))(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(dense_layers - 2):
        x = tf.keras.layers.Dense(units=get_previous_geometric_value(x.shape[-1] - 1, 2.0))(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=get_previous_geometric_value(x.shape[-1] - 1, 2.0))(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(dense_layers - 2, -1, -1):
        x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1] + 1, 2.0))(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=x_flatten.shape[-1])(x)
    x = tf.keras.layers.Reshape(x_input.shape[1:])(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_model_conv(x_train_images):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train_images.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    filters_len = len(filters)

    for i in range(filters_len - 1):
        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same")(x)
            x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding="same")(x)

    for j in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(filters_len - 2, -1, -1):
        x = tf.keras.layers.UpSampling2D()(x)

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same")(x)
            x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=input_shape[-1],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same")(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_model_conv_alex(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])
    x_latent_gaussian_input = tf.keras.Input(shape=latent_shape)

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                         output_dim=filters[-1],
                                         embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
               (x_label_input))
    x_label = x_label[:, 0]

    for i in range(conditional_dense_layers):
        x_label = tf.keras.layers.Dense(units=filters[-1])(x_label)
        x_label = tf.keras.layers.Lambda(tf.keras.activations.relu)(x_label)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b (h1 h2) (w1 w2) c -> b h1 w1 (c h2 w2)",
                                                                "h2": 2,
                                                                "w2": 2})(x)

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None:
        x_res = x

        x = Standardiser()(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Add()([x_res, x])

        x_res = x

        for j in range(conv_layers[-1]):
            x = tf.keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x_latent_mean = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_latent_mean = tf.keras.layers.Conv2D(filters=filters[-1],
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding="same",
                                               kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_mean)

        x_label_beta = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_latent_mean = tf.keras.layers.Multiply()([x_latent_mean, x_label_gamma])
        x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_label_beta])
        x_latent_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_mean)

    x_res = tf.keras.layers.Conv2D(filters=x_latent_mean.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_res])

    x_latent_mean = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_mean)

    x_latent_stddev = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_latent_stddev = tf.keras.layers.Conv2D(filters=filters[-1],
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding="same",
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_stddev)

        x_label_beta = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_latent_stddev = tf.keras.layers.Multiply()([x_latent_stddev, x_label_gamma])
        x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_label_beta])
        x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_stddev)

    x_res = tf.keras.layers.Conv2D(filters=x_latent_stddev.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_res])

    x_latent_stddev = tf.keras.layers.Conv2D(filters=x_latent_gaussian_input.shape[-1],
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_stddev)
    x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_latent_stddev)
    x_latent_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_latent_stddev)

    x = tf.keras.layers.Multiply()([x_latent_gaussian_input, x_latent_stddev])
    x = tf.keras.layers.Add()([x, x_latent_mean])

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    for i in range(filters_len - 2, -1, -1):
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b h w (c1 c2 c3) -> b (h c2) (w c3) c1",
                                                                "c2": 2,
                                                                "c3": 2})(x)
        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.zeros)(x)

    model = tf.keras.Model(inputs=[x_image_input, x_label_input, x_latent_gaussian_input],
                           outputs=[x, x_latent_mean, x_latent_stddev])

    return model


def get_model_conv_alex_discrete(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                         output_dim=filters[-1],
                                         embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
               (x_label_input))
    x_label = x_label[:, 0]

    for i in range(conditional_dense_layers):
        x_label = tf.keras.layers.Dense(units=filters[-1])(x_label)
        x_label = tf.keras.layers.Lambda(tf.keras.activations.relu)(x_label)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b (h1 h2) (w1 w2) c -> b h1 w1 (c h2 w2)",
                                                                "h2": 2,
                                                                "w2": 2})(x)

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None:
        x_res = x

        x = Standardiser()(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Add()([x_res, x])

        x_res = x

        for j in range(conv_layers[-1]):
            x = tf.keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x_shape_prod = tf.math.reduce_prod(x.shape[1:-1]).numpy()
    x_latent_quantised, x_latent_discretised = (
        VectorQuantiser(embedding_dim=x_shape_prod,
                        num_embeddings=int(tf.math.round(x_shape_prod * num_embeddings_multiplier)))(x))

    x = x_latent_quantised

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                               kernel_initializer=tf.keras.initializers.orthogonal)
                         (x_label_gamma))

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    for i in range(filters_len - 2, -1, -1):
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b h w (c1 c2 c3) -> b (h c2) (w c3) c1",
                                                                "c2": 2,
                                                                "c3": 2})(x)
        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.zeros)(x)

    model = tf.keras.Model(inputs=[x_image_input, x_label_input],
                           outputs=[x, x_latent_quantised, x_latent_discretised])

    return model


def get_model_conv_alex_gaussian_negative_log_likelihood(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])
    x_latent_gaussian_input = tf.keras.Input(shape=latent_shape)

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                         output_dim=filters[-1],
                                         embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
               (x_label_input))
    x_label = x_label[:, 0]

    for i in range(conditional_dense_layers):
        x_label = tf.keras.layers.Dense(units=filters[-1])(x_label)
        x_label = tf.keras.layers.Lambda(tf.keras.activations.relu)(x_label)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b (h1 h2) (w1 w2) c -> b h1 w1 (c h2 w2)",
                                                                "h2": 2,
                                                                "w2": 2})(x)

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None:
        x_res = x

        x = Standardiser()(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Add()([x_res, x])

        x_res = x

        for j in range(conv_layers[-1]):
            x = tf.keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x_latent_mean = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_latent_mean = tf.keras.layers.Conv2D(filters=filters[-1],
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding="same",
                                               kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_mean)

        x_label_beta = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_latent_mean = tf.keras.layers.Multiply()([x_latent_mean, x_label_gamma])
        x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_label_beta])
        x_latent_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_mean)

    x_res = tf.keras.layers.Conv2D(filters=x_latent_mean.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_res])

    x_latent_mean = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_mean)

    x_latent_stddev = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_latent_stddev = tf.keras.layers.Conv2D(filters=filters[-1],
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 padding="same",
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_stddev)

        x_label_beta = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_latent_stddev = tf.keras.layers.Multiply()([x_latent_stddev, x_label_gamma])
        x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_label_beta])
        x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_stddev)

    x_res = tf.keras.layers.Conv2D(filters=x_latent_stddev.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_res])

    x_latent_stddev = tf.keras.layers.Conv2D(filters=x_latent_gaussian_input.shape[-1],
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_stddev)
    x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_latent_stddev)
    x_latent_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_latent_stddev)

    x = tf.keras.layers.Multiply()([x_latent_gaussian_input, x_latent_stddev])
    x = tf.keras.layers.Add()([x, x_latent_mean])

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    for i in range(filters_len - 2, -1, -1):
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b h w (c1 c2 c3) -> b (h c2) (w c3) c1",
                                                                "c2": 2,
                                                                "c3": 2})(x)
        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x_mean = x
    x_res = x

    for i in range(conv_layers[0]):
        x_mean = tf.keras.layers.Conv2D(filters=filters[0],
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)

        x_label_beta = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_mean = tf.keras.layers.Multiply()([x_mean, x_label_gamma])
        x_mean = tf.keras.layers.Add()([x_mean, x_label_beta])
        x_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_mean)

    x_res = tf.keras.layers.Conv2D(filters=x_mean.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_mean = tf.keras.layers.Add()([x_mean, x_res])

    x_mean = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=tf.keras.initializers.zeros)(x_mean)

    x_stddev = x
    x_res = x

    for i in range(conv_layers[0]):
        x_stddev = tf.keras.layers.Conv2D(filters=filters[0],
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)

        x_label_beta = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_stddev = tf.keras.layers.Multiply()([x_stddev, x_label_gamma])
        x_stddev = tf.keras.layers.Add()([x_stddev, x_label_beta])
        x_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_stddev)

    x_res = tf.keras.layers.Conv2D(filters=x_stddev.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_stddev = tf.keras.layers.Add()([x_stddev, x_res])

    x_stddev = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)
    x_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_stddev)
    x_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_stddev)

    model = tf.keras.Model(inputs=[x_image_input, x_label_input, x_latent_gaussian_input],
                           outputs=[x_mean, x_stddev, x_latent_mean, x_latent_stddev])

    return model


def get_model_conv_alex_discrete_gaussian_negative_log_likelihood(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                         output_dim=filters[-1],
                                         embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
               (x_label_input))
    x_label = x_label[:, 0]

    for i in range(conditional_dense_layers):
        x_label = tf.keras.layers.Dense(units=filters[-1])(x_label)
        x_label = tf.keras.layers.Lambda(tf.keras.activations.relu)(x_label)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b (h1 h2) (w1 w2) c -> b h1 w1 (c h2 w2)",
                                                                "h2": 2,
                                                                "w2": 2})(x)

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None:
        x_res = x

        x = Standardiser()(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Add()([x_res, x])

        x_res = x

        for j in range(conv_layers[-1]):
            x = tf.keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x_shape_prod = tf.math.reduce_prod(x.shape[1:-1]).numpy()
    x_latent_quantised, x_latent_discretised = (
        VectorQuantiser(embedding_dim=x_shape_prod,
                        num_embeddings=int(tf.math.round(x_shape_prod * num_embeddings_multiplier)))(x))

    x = x_latent_quantised

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                               kernel_initializer=tf.keras.initializers.orthogonal)
                         (x_label_gamma))

        x = tf.keras.layers.Multiply()([x, x_label_gamma])
        x = tf.keras.layers.Add()([x, x_label_beta])
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    for i in range(filters_len - 2, -1, -1):
        x = tf.keras.layers.Lambda(einops.rearrange, arguments={"pattern": "b h w (c1 c2 c3) -> b (h c2) (w c3) c1",
                                                                "c2": 2,
                                                                "c3": 2})(x)
        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
            x_label_beta = tf.keras.layers.Dense(units=x.shape[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
            x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
            x_label_gamma = tf.keras.layers.Dense(units=x.shape[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

            x = tf.keras.layers.Multiply()([x, x_label_gamma])
            x = tf.keras.layers.Add()([x, x_label_beta])
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

    x_mean = x
    x_res = x

    for i in range(conv_layers[0]):
        x_mean = tf.keras.layers.Conv2D(filters=filters[0],
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)

        x_label_beta = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_mean.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_mean = tf.keras.layers.Multiply()([x_mean, x_label_gamma])
        x_mean = tf.keras.layers.Add()([x_mean, x_label_beta])
        x_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_mean)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_mean = tf.keras.layers.Add()([x_mean, x_res])

    x_mean = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=tf.keras.initializers.zeros)(x_mean)

    x_stddev = x
    x_res = x

    for i in range(conv_layers[0]):
        x_stddev = tf.keras.layers.Conv2D(filters=filters[0],
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)

        x_label_beta = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_beta)
        x_label_beta = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_label_beta)

        x_label_gamma = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label)
        x_label_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_label_gamma)
        x_label_gamma = tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                              kernel_initializer=tf.keras.initializers.orthogonal)(x_label_gamma)

        x_stddev = tf.keras.layers.Multiply()([x_stddev, x_label_gamma])
        x_stddev = tf.keras.layers.Add()([x_stddev, x_label_beta])
        x_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_stddev)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x_stddev = tf.keras.layers.Add()([x_stddev, x_res])

    x_stddev = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)
    x_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_stddev)
    x_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_stddev)

    model = tf.keras.Model(inputs=[x_image_input, x_label_input],
                           outputs=[x_mean, x_stddev, x_latent_quantised, x_latent_discretised])

    return model


def sinespace(start, stop, num):
    linspaced = np.linspace(0.0, 90.0, num)

    sinspaced = np.sin(np.deg2rad(linspaced))

    sinspaced_min = np.min(sinspaced)
    sinspaced = start + ((sinspaced - sinspaced_min) * ((stop - start) / ((np.max(sinspaced) - sinspaced_min) +
                                                                          np.finfo(np.float32).eps)))

    return sinspaced


def get_batch_sizes(x_train_images):
    print("get_batch_sizes")

    current_min_batch_size = get_next_geometric_value(min_batch_size, 2.0)

    x_train_images_len = len(x_train_images)

    if max_batch_size is not None:
        if max_batch_size < current_min_batch_size or max_batch_size > x_train_images_len:
            current_max_batch_size = x_train_images_len
        else:
            current_max_batch_size = max_batch_size
    else:
        current_max_batch_size = x_train_images_len

    batch_sizes = [current_min_batch_size]

    while True:
        current_batch_size = int(np.round(batch_sizes[-1] * 2.0))

        if current_batch_size <= current_max_batch_size:
            batch_sizes.append(current_batch_size)
        else:
            break

    if current_max_batch_size >= x_train_images_len > batch_sizes[-1]:
        batch_sizes.append(x_train_images_len)

    batch_sizes_epochs = sinespace(0.0, epochs - 1, len(batch_sizes) + 1)
    batch_sizes_epochs = np.round(batch_sizes_epochs)

    return batch_sizes, batch_sizes_epochs


def flip_image(image):
    if axis_zero_flip_bool:
        if random.choice([True, False]):
            image = np.flip(image, axis=0)

    if axis_one_flip_bool:
        if random.choice([True, False]):
            image = np.flip(image, axis=1)

    return image


def translate_image(image):
    max_translation = int(np.round(image.shape[0] * translate_proportion))

    translation = random.randint(0, max_translation)

    if random.choice([True, False]):
        image = np.pad(image, ((0, translation), (0, 0), (0, 0)))  # noqa
    else:
        image = np.pad(image, ((translation, 0), (0, 0), (0, 0)))  # noqa

    translation = random.randint(0, max_translation)

    if random.choice([True, False]):
        image = np.pad(image, ((0, 0), (0, translation), (0, 0)))  # noqa
    else:
        image = np.pad(image, ((0, 0), (translation, 0), (0, 0)))  # noqa

    return image


def crop_image(image, output_dimension_size):
    while image.shape[0] - 1 > output_dimension_size:
        image = image[1:-1]

    if image.shape[0] > output_dimension_size:
        image = image[1:]

    while image.shape[1] - 1 > output_dimension_size:
        image = image[:, 1:-1]

    if image.shape[1] > output_dimension_size:
        image = image[:, 1:]

    return image


def augmentation(image):
    image = image.numpy()

    input_dimension_size = image.shape[0]

    image = flip_image(image)

    if gaussian_bool:
        image = gaussian(image, sigma=random.uniform(0.0, max_sigma), mode="constant", preserve_range=True,
                         channel_axis=-1)

    if sharpen_bool:
        image = unsharp_mask(image, radius=random.uniform(0.0, max_radius), amount=random.uniform(0.0, max_amount),
                             preserve_range=True, channel_axis=1)

    if scale_bool:
        image = rescale(image, random.uniform(min_scale, max_scale), order=3, preserve_range=True, channel_axis=-1)

    if rotate_bool:
        image = scipy.ndimage.rotate(image, angle=random.uniform(min_angle, max_angle), axes=(0, 1), order=1)

    if translate_bool:
        image = translate_image(image)

    image = pad_image(image, input_dimension_size)
    image = crop_image(image, input_dimension_size)

    image = tf.convert_to_tensor(image)

    return image


def mean_squared_error(y_true, y_pred):
    loss = tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2.0))

    return loss


def root_mean_squared_error(y_true, y_pred):
    loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2.0)) + tf.keras.backend.epsilon())

    return loss


def gaussian_negative_log_likelihood(y_true, y_pred_mean, y_pred_std):
    y_true = tf.reshape(y_true, (y_true.shape[0], -1))
    y_pred_mean = tf.reshape(y_pred_mean, (y_pred_mean.shape[0], -1))
    y_pred_std = tf.reshape(y_pred_std, (y_pred_std.shape[0], -1))

    loss = (
        -tf.math.reduce_mean((-(tf.math.pow((tf.math.divide_no_nan((y_true - y_pred_mean), y_pred_std)), 2.0) / 2.0)
                              - (tf.math.log(y_pred_std + tf.keras.backend.epsilon())) -  # noqa
                              (tf.math.log(2.0 * np.pi) / 2.0))))  # noqa

    return loss


def gaussian_kullback_leibler_divergence(y_pred_mean, y_pred_std):
    y_pred_mean = tf.reshape(y_pred_mean, (y_pred_mean.shape[0], -1))
    y_pred_std = tf.reshape(y_pred_std, (y_pred_std.shape[0], -1))

    loss = tf.math.reduce_mean((tf.math.log(tf.math.divide_no_nan(1.0, y_pred_std) + tf.keras.backend.epsilon()) +  # noqa
                                ((tf.math.pow(y_pred_std, 2.0) + tf.math.pow(y_pred_mean, 2.0)) / 2.0)) -
                               (1.0 / 2.0))

    return loss


def get_error(y_true, y_pred):
    error = tf.math.reduce_mean(tf.math.abs(tf.math.log(tf.math.abs(tf.math.divide_no_nan(y_pred, y_true)) +
                                                        tf.keras.backend.epsilon()))) * 100.0

    return error


def output_image(image, standard_scaler, current_output_path):
    image = image.numpy()

    image = np.reshape(standard_scaler.inverse_transform(np.reshape(image, (-1, 1))), image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    if greyscale_bool:
        image = image[:, :, 0]

    image = Image.fromarray(image)
    image.save(current_output_path)

    return True


def validate(model, x_test_images, standard_scaler, x_test_labels, i):
    validate_output_path = "{0}/validate/".format(output_path)

    output_input_bool = False

    if not os.path.exists(validate_output_path):
        mkdir_p(validate_output_path)

        output_input_bool = True

    current_x_test_image = get_data_from_storage(x_test_images[0])

    current_x_test_label = x_test_labels[0]

    current_x_test_image = tf.expand_dims(current_x_test_image, axis=0)
    current_x_test_label = tf.expand_dims(current_x_test_label, axis=0)

    if conv_bool and alex_bool:
        if gaussian_negative_log_likelihood_bool:
            if discrete_bool:
                y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                    model([current_x_test_image, current_x_test_label], training=False))

                current_y_pred_mean = y_pred_mean[0]
                output_image(current_y_pred_mean, standard_scaler, "{0}/{1}_mean.png".format(validate_output_path,
                                                                                             str(i)))

                y_pred = []

                for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                    y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                y_pred = tf.math.reduce_mean(y_pred, axis=0)
            else:
                y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                    model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                          training=False))

                y_pred_means = [y_pred_mean]
                y_pred_stddevs = [y_pred_stddev]

                for j in range(
                        int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0), 32.0])) - 1):
                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                              training=False))

                    y_pred_means.append(y_pred_mean)
                    y_pred_stddevs.append(y_pred_stddev)

                y_pred_mean = tf.math.reduce_mean(y_pred_means, axis=0)
                y_pred_stddev = tf.math.reduce_mean(y_pred_stddevs, axis=0)

                current_y_pred_mean = y_pred_mean[0]
                output_image(current_y_pred_mean, standard_scaler, "{0}/{1}_mean.png".format(validate_output_path,
                                                                                             str(i)))

                y_pred = []

                for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                    y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                y_pred = tf.math.reduce_mean(y_pred, axis=0)
        else:
            if discrete_bool:
                y_pred, x_latent_quantised, x_latent_discretised = model([current_x_test_image, current_x_test_label],
                                                                         training=False)
            else:
                y_pred, y_latent_mean, y_latent_stddev = (
                    model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                          training=False))

                y_preds = [y_pred]

                for j in range(
                        int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0), 32.0])) - 1):
                    y_pred, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                              training=False))

                    y_preds.append(y_pred)

                y_pred = tf.math.reduce_mean(y_preds, axis=0)
    else:
        y_pred = model([current_x_test_image], training=False)

    current_y_pred = y_pred[0]
    output_image(current_y_pred, standard_scaler, "{0}/{1}.png".format(validate_output_path, str(i)))

    if output_input_bool:
        output_image(current_x_test_image[0], standard_scaler, "{0}/input.png".format(validate_output_path))

    return True


def train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images,
                                x_train_padding_masks, x_test_images, standard_scaler, x_train_labels, x_test_labels):
    print("train")

    validate(model, x_test_images, standard_scaler, x_test_labels, 0)

    x_train_images_len = len(x_train_images)

    current_batch_size = None
    indices = list(range(x_train_images_len))

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    for i in range(epochs):
        for j in range(batch_sizes_epochs_len - 1, 0, -1):
            if batch_sizes_epochs[j - 1] <= i < batch_sizes_epochs[j]:
                current_batch_size = batch_sizes[j - 1]

                break

        iterations = int(np.floor(x_train_images_len / current_batch_size))

        random.shuffle(indices)

        for j in range(iterations):
            accumulated_gradients = [tf.zeros_like(trainable_variable) for trainable_variable in
                                     model.trainable_variables]

            current_index = current_batch_size * j

            losses = []
            errors = []
            errors_range = []

            for m in range(current_batch_size):
                current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])

                current_x_train_image = augmentation(current_x_train_image)
                current_x_train_padding_mask = get_data_from_storage(x_train_padding_masks[indices[current_index]])
                current_x_train_label = x_train_labels[indices[current_index]]

                current_x_train_image = tf.expand_dims(current_x_train_image, axis=0)
                current_x_train_label = tf.expand_dims(current_x_train_label, axis=0)

                if conv_bool and alex_bool:
                    if gaussian_negative_log_likelihood_bool:
                        if discrete_bool:
                            if i + 1 > mean_squared_error_epoch:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                        model([current_x_train_image, current_x_train_label], training=True))

                                    y_pred_mean = y_pred_mean * current_x_train_padding_mask
                                    y_pred_stddev = y_pred_stddev * current_x_train_padding_mask

                                    loss = tf.math.reduce_sum([
                                        gaussian_negative_log_likelihood(current_x_train_image, y_pred_mean,
                                                                         y_pred_stddev),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                        model([current_x_train_image, current_x_train_label], training=True))

                                    y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                    y_pred = y_pred * current_x_train_padding_mask

                                    loss = tf.math.reduce_sum([mean_squared_error(current_x_train_image, y_pred),
                                                               tf.math.reduce_sum(model.losses)])
                        else:
                            if i + 1 > mean_squared_error_epoch:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_image, current_x_train_label,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    y_pred_mean = y_pred_mean * current_x_train_padding_mask
                                    y_pred_stddev = y_pred_stddev * current_x_train_padding_mask

                                    loss = tf.math.reduce_sum([
                                        gaussian_negative_log_likelihood(current_x_train_image, y_pred_mean,
                                                                         y_pred_stddev),
                                        gaussian_latent_loss_weight *
                                        gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_image, current_x_train_label,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                    y_pred = y_pred * current_x_train_padding_mask

                                    loss = tf.math.reduce_mean([
                                        mean_squared_error(current_x_train_image, y_pred),
                                        gaussian_latent_loss_weight * mean_squared_error(tf.zeros(y_latent_mean.shape),
                                                                                         y_latent_mean),
                                        gaussian_latent_loss_weight * mean_squared_error(tf.ones(y_latent_stddev.shape),
                                                                                         y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])

                        y_pred_stddev_full_coverage = y_pred_stddev * 3.0

                        error_bounds = [get_error(current_x_train_image, y_pred_mean + y_pred_stddev_full_coverage),
                                        get_error(current_x_train_image, y_pred_mean - y_pred_stddev_full_coverage)]
                        error_upper_bound = tf.reduce_max(error_bounds)
                        error_lower_bound = tf.reduce_min(error_bounds)

                        errors.append(tf.math.reduce_mean([error_upper_bound, error_lower_bound]))
                        errors_range.append(error_upper_bound - error_lower_bound)
                    else:
                        if discrete_bool:
                            if i + 1 > mean_squared_error_epoch:
                                with tf.GradientTape() as tape:
                                    y_pred, x_latent_quantised, x_latent_discretised = (
                                        model([current_x_train_image, current_x_train_label], training=True))

                                    y_pred = y_pred * current_x_train_padding_mask

                                    loss = tf.math.reduce_sum([root_mean_squared_error(current_x_train_image, y_pred),
                                                               tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred, x_latent_quantised, x_latent_discretised = (
                                        model([current_x_train_image, current_x_train_label], training=True))

                                    y_pred = y_pred * current_x_train_padding_mask

                                    loss = tf.math.reduce_sum([mean_squared_error(current_x_train_image, y_pred),
                                                               tf.math.reduce_sum(model.losses)])
                        else:
                            if i + 1 > mean_squared_error_epoch:
                                with tf.GradientTape() as tape:
                                    y_pred, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_image, current_x_train_label,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    y_pred = y_pred * current_x_train_padding_mask

                                    loss = tf.math.reduce_sum([
                                        root_mean_squared_error(current_x_train_image, y_pred),
                                        gaussian_latent_loss_weight *
                                        gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_image, current_x_train_label,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    y_pred = y_pred * current_x_train_padding_mask

                                    loss = tf.math.reduce_mean([
                                        mean_squared_error(current_x_train_image, y_pred),
                                        gaussian_latent_loss_weight * mean_squared_error(tf.zeros(y_latent_mean.shape),
                                                                                         y_latent_mean),
                                        gaussian_latent_loss_weight * mean_squared_error(tf.ones(y_latent_stddev.shape),
                                                                                         y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])

                        errors.append(get_error(current_x_train_image, y_pred))
                        errors_range.append(tf.constant(0.0))
                else:
                    with tf.GradientTape() as tape:
                        y_pred = model([current_x_train_image], training=True)

                        y_pred = y_pred * current_x_train_padding_mask

                        loss = tf.math.reduce_sum([mean_squared_error(current_x_train_image, y_pred),
                                                   tf.math.reduce_sum(model.losses)])

                    errors.append(get_error(current_x_train_image, y_pred))
                    errors_range.append(tf.constant(0.0))

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)

                current_index = current_index + 1

            gradients = [gradient / current_batch_size for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            error = tf.math.reduce_mean(errors)
            error_range = tf.math.reduce_mean(errors_range)

            if alex_bool and i + 1 > mean_squared_error_epoch:
                if gaussian_negative_log_likelihood_bool:
                    loss_name = "Gaussian NLL"
                else:
                    loss_name = "RMSE"
            else:
                loss_name = "MSE"

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss ({5}): {6:14} Error: {7:14} +/- {8:14}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), loss_name,
                str(loss.numpy()), str(error.numpy()), str(error_range.numpy())))

        validate(model, x_test_images, standard_scaler, x_test_labels, i + 1)

    return model


def train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images, x_train_padding_masks, x_test_images,
          standard_scaler, x_train_labels, x_test_labels):
    print("train")

    validate(model, x_test_images, standard_scaler, x_test_labels, 0)

    x_train_images_len = len(x_train_images)

    current_batch_size = None
    indices = list(range(x_train_images_len))

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    for i in range(epochs):
        for j in range(batch_sizes_epochs_len - 1, 0, -1):
            if batch_sizes_epochs[j - 1] <= i <= batch_sizes_epochs[j]:
                current_batch_size = batch_sizes[j - 1]

                break

        iterations = int(np.floor(x_train_images_len / current_batch_size))

        random.shuffle(indices)

        for j in range(iterations):
            current_index = current_batch_size * j

            current_x_train_images = []
            current_x_train_padding_masks = []
            current_x_train_labels = []

            for k in range(current_batch_size):
                current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])

                current_x_train_images.append(augmentation(current_x_train_image))
                current_x_train_padding_masks.append(get_data_from_storage(
                    x_train_padding_masks[indices[current_index]]))
                current_x_train_labels.append(x_train_labels[indices[current_index]])

                current_index = current_index + 1

            current_x_train_images = tf.convert_to_tensor(current_x_train_images)
            current_x_train_padding_masks = tf.convert_to_tensor(current_x_train_padding_masks)
            current_x_train_labels = tf.convert_to_tensor(current_x_train_labels)

            if conv_bool and alex_bool:
                if gaussian_negative_log_likelihood_bool:
                    if discrete_bool:
                        if i + 1 > mean_squared_error_epoch:
                            with tf.GradientTape() as tape:
                                y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                    model([current_x_train_images, current_x_train_labels], training=True))

                                y_pred_mean = y_pred_mean * current_x_train_padding_masks
                                y_pred_stddev = y_pred_stddev * current_x_train_padding_masks

                                loss = tf.math.reduce_sum([
                                    gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                     y_pred_stddev),
                                    tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                    model([current_x_train_images, current_x_train_labels], training=True))

                                y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                y_pred = y_pred * current_x_train_padding_masks

                                loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                    else:
                        if i + 1 > mean_squared_error_epoch:
                            with tf.GradientTape() as tape:
                                y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)],
                                          training=True))

                                y_pred_mean = y_pred_mean * current_x_train_padding_masks
                                y_pred_stddev = y_pred_stddev * current_x_train_padding_masks

                                loss = tf.math.reduce_sum([
                                    gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                     y_pred_stddev),
                                    gaussian_latent_loss_weight * gaussian_kullback_leibler_divergence(y_latent_mean,
                                                                                                       y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)], training=True))

                                y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                y_pred = y_pred * current_x_train_padding_masks

                                loss = tf.math.reduce_mean([
                                    mean_squared_error(current_x_train_images, y_pred),
                                    gaussian_latent_loss_weight * mean_squared_error(tf.zeros(y_latent_mean.shape),
                                                                                     y_latent_mean),
                                    gaussian_latent_loss_weight * mean_squared_error(tf.ones(y_latent_stddev.shape),
                                                                                     y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])

                    y_pred_stddev_full_coverage = y_pred_stddev * 3.0

                    error_bounds = [get_error(current_x_train_images, y_pred_mean + y_pred_stddev_full_coverage),
                                    get_error(current_x_train_images, y_pred_mean - y_pred_stddev_full_coverage)]
                    error_upper_bound = tf.reduce_max(error_bounds)
                    error_lower_bound = tf.reduce_min(error_bounds)

                    error = get_error(current_x_train_images, y_pred_mean)
                    error_range = error_upper_bound - error_lower_bound
                else:
                    if discrete_bool:
                        if i + 1 > mean_squared_error_epoch:
                            with tf.GradientTape() as tape:
                                y_pred, x_latent_quantised, x_latent_discretised = (
                                    model([current_x_train_images, current_x_train_labels], training=True))

                                y_pred = y_pred * current_x_train_padding_masks

                                loss = tf.math.reduce_sum([root_mean_squared_error(current_x_train_images, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred, x_latent_quantised, x_latent_discretised = (
                                    model([current_x_train_images, current_x_train_labels], training=True))

                                y_pred = y_pred * current_x_train_padding_masks

                                loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                    else:
                        if i + 1 > mean_squared_error_epoch:
                            with tf.GradientTape() as tape:
                                y_pred, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)],
                                          training=True))

                                y_pred = y_pred * current_x_train_padding_masks

                                loss = tf.math.reduce_sum([
                                    root_mean_squared_error(current_x_train_images, y_pred),
                                    gaussian_latent_loss_weight * gaussian_kullback_leibler_divergence(y_latent_mean,
                                                                                                       y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)], training=True))

                                y_pred = y_pred * current_x_train_padding_masks

                                loss = tf.math.reduce_mean([
                                    mean_squared_error(current_x_train_images, y_pred),
                                    gaussian_latent_loss_weight * mean_squared_error(tf.zeros(y_latent_mean.shape),
                                                                                     y_latent_mean),
                                    gaussian_latent_loss_weight * mean_squared_error(tf.ones(y_latent_stddev.shape),
                                                                                     y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])

                    error = get_error(current_x_train_images, y_pred)
                    error_range = tf.constant(0.0)
            else:
                with tf.GradientTape() as tape:
                    y_pred = model([current_x_train_images], training=True)

                    y_pred = y_pred * current_x_train_padding_masks

                    loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred),
                                               tf.math.reduce_sum(model.losses)])

                error = get_error(current_x_train_images, y_pred)
                error_range = tf.constant(0.0)

            gradients = tape.gradient(loss, model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            if alex_bool and i + 1 > mean_squared_error_epoch:
                if gaussian_negative_log_likelihood_bool:
                    loss_name = "Gaussian NLL"
                else:
                    loss_name = "RMSE"
            else:
                loss_name = "MSE"

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss ({5}): {6:14} Error: {7:14} +/- {8:14}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), loss_name,
                str(loss.numpy()), str(error.numpy()), str(error_range.numpy())))

        validate(model, x_test_images, standard_scaler, x_test_labels, i + 1)

    return model


def test(model, x_test_images, standard_scaler, x_test_labels):
    print("test")

    test_output_path = "{0}/test/".format(output_path)
    mkdir_p(test_output_path)

    errors = []

    for i in range(int(np.floor(len(x_test_images) / test_batch_size))):
        current_x_test_images = []
        current_x_test_labels = []

        index = i * test_batch_size

        for j in range(test_batch_size):
            current_index = index + j

            current_x_test_image = get_data_from_storage(x_test_images[current_index])

            current_x_test_images.append(current_x_test_image)
            current_x_test_labels.append(x_test_labels[current_index])

        current_x_test_images = tf.convert_to_tensor(current_x_test_images)
        current_x_test_labels = tf.convert_to_tensor(current_x_test_labels)

        if conv_bool and alex_bool:
            if gaussian_negative_log_likelihood_bool:
                if discrete_bool:
                    y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                        model([current_x_test_images, current_x_test_labels], training=False))

                    for j in range(test_batch_size):
                        current_index = index + j

                        current_y_pred_mean = y_pred_mean[j]
                        output_image(current_y_pred_mean, standard_scaler,
                                     "{0}/{1}_mean.png".format(test_output_path, str(current_index)))

                    y_pred = []

                    for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                        y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                    y_pred = tf.math.reduce_mean(y_pred, axis=0)
                else:
                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_images, current_x_test_labels,
                               tf.random.normal((test_batch_size,) + latent_shape)], training=False))

                    y_pred_means = [y_pred_mean]
                    y_pred_stddevs = [y_pred_stddev]

                    for j in range(int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0),
                                                           32.0])) - 1):
                        y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                            model([current_x_test_images, current_x_test_labels,
                                   tf.random.normal((1,) + latent_shape)], training=False))

                        y_pred_means.append(y_pred_mean)
                        y_pred_stddevs.append(y_pred_stddev)

                    y_pred_mean = tf.math.reduce_mean(y_pred_means, axis=0)
                    y_pred_stddev = tf.math.reduce_mean(y_pred_stddevs, axis=0)

                    for j in range(test_batch_size):
                        current_index = index + j

                        current_y_pred_mean = y_pred_mean[j]
                        output_image(current_y_pred_mean, standard_scaler,
                                     "{0}/{1}_mean.png".format(test_output_path, str(current_index)))

                    y_pred = []

                    for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                        y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                    y_pred = tf.math.reduce_mean(y_pred, axis=0)
            else:
                if discrete_bool:
                    y_pred, x_latent_quantised, x_latent_discretised = (
                        model([current_x_test_images, current_x_test_labels], training=False))
                else:
                    y_pred, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_images, current_x_test_labels,
                               tf.random.normal((test_batch_size,) + latent_shape)], training=False))

                    y_preds = [y_pred]

                    for j in range(int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0),
                                                           32.0])) - 1):
                        y_pred, y_latent_mean, y_latent_stddev = (
                            model([current_x_test_images, current_x_test_labels,
                                   tf.random.normal((1,) + latent_shape)], training=False))

                        y_preds.append(y_pred)

                    y_pred = tf.math.reduce_mean(y_preds, axis=0)
        else:
            y_pred = model([current_x_test_images], training=False)

        errors.append(get_error(current_x_test_images, y_pred))

        for j in range(test_batch_size):
            current_index = index + j

            current_y_pred = y_pred[j]
            output_image(current_y_pred, standard_scaler, "{0}/{1}.png".format(test_output_path,
                                                                               str(current_index)))

            current_x_test_image = current_x_test_images[j]
            output_image(current_x_test_image, standard_scaler, "{0}/{1}_input.png".format(test_output_path,
                                                                                           str(current_index)))

    error = tf.math.reduce_mean(errors)

    print("Error: {0:14}%".format(str(error.numpy())))

    return True


def main():
    print("main")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if read_data_from_storage_bool:
        mkdir_p(output_path)

    x_train_images, x_test_images, x_train_labels, x_test_labels = get_input()
    x_train_images, x_train_padding_masks, x_test_images, standard_scaler, x_train_labels, x_test_labels = (
        preprocess_input(x_train_images, x_test_images, x_train_labels, x_test_labels))

    if conv_bool:
        if alex_bool:
            if gaussian_negative_log_likelihood_bool:
                if discrete_bool:
                    model = get_model_conv_alex_discrete_gaussian_negative_log_likelihood(x_train_images,
                                                                                          x_train_labels)
                else:
                    model = get_model_conv_alex_gaussian_negative_log_likelihood(x_train_images, x_train_labels)
            else:
                if discrete_bool:
                    model = get_model_conv_alex_discrete(x_train_images, x_train_labels)
                else:
                    model = get_model_conv_alex(x_train_images, x_train_labels)
        else:
            model = get_model_conv(x_train_images)
    else:
        if deep_bool:
            model = get_model_deep_dense(x_train_images)
        else:
            model = get_model_dense(x_train_images)

    model.summary()

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         weight_decay=weight_decay,
                                         use_ema=True,
                                         ema_overwrite_frequency=ema_overwrite_frequency)

    batch_sizes, batch_sizes_epochs = get_batch_sizes(x_train_images)

    if gradient_accumulation_bool:
        model = train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images,
                                            x_train_padding_masks, x_test_images, standard_scaler, x_train_labels,
                                            x_test_labels)
    else:
        model = train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images, x_train_padding_masks,
                      x_test_images, standard_scaler, x_train_labels, x_test_labels)

    if epochs > 0:
        optimiser.finalize_variable_values(model.trainable_variables)

    test(model, x_test_images, standard_scaler, x_test_labels)

    return True


if __name__ == "__main__":
    main()
