# Copyright University College London 2023, 2024
# Author: Alexander C. Whitehead, Department of Computer Science, UCL
# For internal research only.


import os
import shutil
import errno
import random
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
from positional_encodings.tf_encodings import TFPositionalEncoding1D
from standardiser import Standardiser
import pickle
from PIL import Image


np.seterr(all="print")


dataset_name = "mnist"
output_path = "../output/generation/"

read_data_from_storage_bool = False

preprocess_list_bool = False
greyscale_bool = True

alex_bool = True
image_input_bool = True

if alex_bool:
    min_output_dimension_size = 4
    max_output_dimension_size = 32

    conditional_dense_layers = 2
    filters = [64, 128, 256, 512, 1024]
    conv_layers = [2, 2, 2, 2, 2]
    num_heads = [4, 4, 4, 4, 4]
    key_dim = [32, 32, 32, 32, 32]
    image_input_bottleneck_units_divider = 2
else:
    min_output_dimension_size = 4
    max_output_dimension_size = 32

    filters = [64, 128, 256, 512]
    num_heads = 4
    key_dim = 32

if alex_bool:
    learning_rate = 1e-04
    weight_decay = 0.0
    ema_overwrite_frequency = None
else:
    learning_rate = 1e-04
    ema_overwrite_frequency = 10

gradient_accumulation_bool = False

epochs = 256

if gradient_accumulation_bool:
    if alex_bool:
        min_batch_size = 32
        max_batch_size = 32
    else:
        min_batch_size = 16
        max_batch_size = 16
else:
    if alex_bool:
        min_batch_size = 32
        max_batch_size = 32
    else:
        min_batch_size = 16
        max_batch_size = 16

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

number_of_timesteps = 1000

unbatch_timesteps_bool = True

if alex_bool:
    mean_squared_error_epochs = 1
else:
    mean_squared_error_epochs = epochs

output_start_timestep_proportion = 1.0
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


def get_next_geometric_value(an, a0):
    n = np.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2.0, (np.ceil(n) - 1.0))

    return an


def get_positional_encodings(shape):
    print("get_positional_encodings")

    positional_encodings = []

    for i in range(number_of_timesteps):
        positional_encodings.append(np.zeros(shape))

    positional_encodings = np.array(positional_encodings)
    positional_encodings = np.expand_dims(positional_encodings, axis=0)
    positional_encodings = tf.convert_to_tensor(positional_encodings)
    positional_encodings = einops.rearrange(positional_encodings, "b h c -> b c h")

    positional_encodings_layer = TFPositionalEncoding1D(number_of_timesteps)
    positional_encodings = positional_encodings_layer(positional_encodings)

    positional_encodings = einops.rearrange(positional_encodings, "b c h -> b h c")
    positional_encodings = positional_encodings.numpy()
    positional_encodings = positional_encodings[0]

    return positional_encodings


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
        if output_dimension_size > max_output_dimension_size:  # noqa
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

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        image = pad_image(image, output_dimension_size)

        images[i] = set_data_from_storage(image, images[i])

    return images


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

    x_train_images = pad_images_list(x_train_images)
    x_test_images = pad_images_list(x_test_images)

    x_train_images = convert_images_to_tensor_list(x_train_images)
    x_test_images = convert_images_to_tensor_list(x_test_images)

    return x_train_images, x_test_images, standard_scaler


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

    x_train_images = pad_images_array(x_train_images)
    x_test_images = pad_images_array(x_test_images)

    x_train_images = tf.convert_to_tensor(x_train_images)
    x_test_images = tf.convert_to_tensor(x_test_images)

    return x_train_images, x_test_images, standard_scaler


def preprocess_positional_encodings(x_positional_encodings):
    print("preprocess_positional_encodings")

    x_positional_encodings = np.reshape(StandardScaler().fit_transform(
        np.reshape(x_positional_encodings, (-1, 1))), x_positional_encodings.shape)

    x_positional_encodings = tf.convert_to_tensor(x_positional_encodings)

    return x_positional_encodings


def preprocess_labels(x_train_labels, x_test_labels):
    print("preprocess_labels")

    x_train_labels = tf.convert_to_tensor(x_train_labels)
    x_test_labels = tf.convert_to_tensor(x_test_labels)

    return x_train_labels, x_test_labels


def preprocess_input(x_train_images, x_test_images, x_timestep_encodings, x_train_labels, x_test_labels):
    print("preprocess_input")

    if greyscale_bool:
        x_train_images = convert_rgb_to_greyscale(x_train_images)
        x_test_images = convert_rgb_to_greyscale(x_test_images)

    if read_data_from_storage_bool or preprocess_list_bool:
        x_train_images, x_test_images, standard_scaler = preprocess_images_list(x_train_images, x_test_images)
    else:
        x_train_images, x_test_images, standard_scaler = preprocess_images_array(x_train_images, x_test_images)

    x_timestep_encodings = preprocess_positional_encodings(x_timestep_encodings)

    x_train_labels, x_test_labels = preprocess_labels(x_train_labels, x_test_labels)

    return x_train_images, x_test_images, standard_scaler, x_timestep_encodings, x_train_labels, x_test_labels


def get_previous_geometric_value(an, a0):
    n = np.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2.0, (np.floor(n) - 1.0))

    return an


def get_model_conv(x_train_images, x_timestep_encodings, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_input = tf.keras.Input(shape=image_input_shape)
    x_timestep_encoding_input = tf.keras.Input(shape=x_timestep_encodings.shape[1:])
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    x = x_input

    x_shape_prod = tf.math.reduce_prod(x.shape[1:-1]).numpy()
    x_label = tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                        output_dim=x_shape_prod)(x_label_input)
    x_label = x_label[:, 0]
    x_label = tf.keras.layers.Reshape(target_shape=x.shape[1:])(x_label)

    x = tf.keras.layers.Concatenate()([x, x_label])

    x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same")(x)

    x_timestep_encoding = x_timestep_encoding_input
    x_timestep_encoding = tf.keras.layers.Dense(units=filters[-1])(x_timestep_encoding)
    x_timestep_encoding = tf.keras.layers.Lambda(tf.keras.activations.gelu)(x_timestep_encoding)
    x_timestep_encoding = tf.keras.layers.Dense(units=filters[-1])(x_timestep_encoding)

    filters_len = len(filters)

    x_skips = [x]

    filters_len_minus_one = filters_len - 1
    filters_len_minus_two = filters_len_minus_one - 1

    for i in range(filters_len_minus_one):
        x_res = x

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)

        x = tf.keras.layers.GroupNormalization(groups=8)(x)

        x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
        x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
        x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                          (x_timestep_encoding_beta_gamma))

        x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
        x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

        x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

        x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
        x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape != x.shape:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same")(x_res)

        x = tf.keras.layers.Add()([x, x_res])

        x_skips.append(x)

        x_res = x

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)

        x = tf.keras.layers.GroupNormalization(groups=8)(x)

        x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
        x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
        x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                          (x_timestep_encoding_beta_gamma))

        x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
        x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

        x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

        x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
        x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape != x.shape:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same")(x_res)

        x = tf.keras.layers.Add()([x, x_res])

        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1,
                                               center=False)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                           key_dim=key_dim,
                                           feature_transform="elu")(x, x)
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape != x.shape:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same")(x_res)

        x = tf.keras.layers.Add()([x, x_res])

        x_skips.append(x)

        if i < filters_len_minus_two:
            x = tf.keras.layers.Lambda(einops.rearrange,
                                       arguments={"pattern": "b (h1 h2) (w1 w2) c -> b h1 w1 (c h2 w2)",
                                                  "h2": 2,
                                                  "w2": 2})(x)

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)

    x_res = x

    x = tf.keras.layers.Conv2D(filters=filters[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)

    x = tf.keras.layers.GroupNormalization(groups=8)(x)

    x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
    x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
    x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                      (x_timestep_encoding_beta_gamma))

    x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
    x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

    x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

    x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
    x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=8)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape != x.shape:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same")(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    x_res = x

    x = tf.keras.layers.GroupNormalization(groups=1,
                                           center=False)(x)

    x_shape = x.shape
    x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
    x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                           key_dim=key_dim)(x, x)
    x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

    if x_res.shape != x.shape:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same")(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    x_res = x

    x = tf.keras.layers.Conv2D(filters=filters[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)

    x = tf.keras.layers.GroupNormalization(groups=8)(x)

    x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
    x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
    x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                      (x_timestep_encoding_beta_gamma))

    x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
    x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

    x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

    x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
    x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=8)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape != x.shape:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same")(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    for i in range(filters_len - 2, -1, -1):
        x = tf.keras.layers.Concatenate()([x, x_skips.pop()])

        x_res = x

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)

        x = tf.keras.layers.GroupNormalization(groups=8)(x)

        x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
        x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
        x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                          (x_timestep_encoding_beta_gamma))

        x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
        x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

        x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

        x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
        x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape != x.shape:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same")(x_res)

        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.Concatenate()([x, x_skips.pop()])

        x_res = x

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)

        x = tf.keras.layers.GroupNormalization(groups=8)(x)

        x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
        x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
        x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                          (x_timestep_encoding_beta_gamma))

        x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
        x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

        x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

        x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
        x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x = tf.keras.layers.Conv2D(filters=filters[i],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.GroupNormalization(groups=8)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape != x.shape:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same")(x_res)

        x = tf.keras.layers.Add()([x, x_res])

        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1,
                                               center=False)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                           key_dim=key_dim,
                                           feature_transform="elu")(x, x)
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape != x.shape:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same")(x_res)

        x = tf.keras.layers.Add()([x, x_res])

        if i > 0:
            x = tf.keras.layers.UpSampling2D()(x)

        x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)

    x = tf.keras.layers.Concatenate()([x, x_skips.pop()])

    x_res = x

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)

    x = tf.keras.layers.GroupNormalization(groups=8)(x)

    x_timestep_encoding_beta_gamma_units = int(np.round(x.shape[-1] * 2.0))
    x_timestep_encoding_beta_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_timestep_encoding)
    x_timestep_encoding_beta_gamma = (tf.keras.layers.Dense(units=x_timestep_encoding_beta_gamma_units)
                                      (x_timestep_encoding_beta_gamma))

    x_timestep_encoding_gamma = x_timestep_encoding_beta_gamma[:, :x.shape[-1]]
    x_timestep_encoding_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_timestep_encoding_gamma)

    x_timestep_encoding_beta = x_timestep_encoding_beta_gamma[:, x.shape[-1]:]

    x = tf.keras.layers.Multiply()([x, x_timestep_encoding_gamma])
    x = tf.keras.layers.Add()([x, x_timestep_encoding_beta])

    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same")(x)
    x = tf.keras.layers.GroupNormalization(groups=8)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape != x.shape:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same")(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    x = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same")(x)

    model = tf.keras.Model(inputs=[x_input, x_timestep_encoding_input, x_label_input],
                           outputs=[x])

    return model


def get_model_conv_alex(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_noise_input = tf.keras.Input(shape=image_input_shape)
    x_timestep_encoding_input = tf.keras.Input(shape=1)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_noise_input)

    x_timestep_encoding = (
        tf.keras.layers.Embedding(input_dim=number_of_timesteps,
                                  output_dim=filters[-1],
                                  embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
        (x_timestep_encoding_input))
    x_timestep_encoding = x_timestep_encoding[:, 0]
    x_timestep_encoding = (tf.keras.layers.Dense(units=filters[-1],
                                                 kernel_initializer=tf.keras.initializers.orthogonal)
                           (x_timestep_encoding))

    x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                         output_dim=filters[-1],
                                         embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
               (x_label_input))
    x_label = x_label[:, 0]
    x_label = (tf.keras.layers.Dense(units=filters[-1],
                                     kernel_initializer=tf.keras.initializers.orthogonal)
               (x_label))

    x_conditioning = tf.keras.layers.Concatenate()([x_timestep_encoding, x_label])

    for i in range(conditional_dense_layers):
        x_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                kernel_initializer=tf.keras.initializers.orthogonal)
                          (x_conditioning))
        x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    x_skips = []

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning))
            x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning_beta))

            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning))
            x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning_gamma))

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x_skip = x

        if num_heads[i] is not None:
            x_res = x_skip

            x = Standardiser()(x)

            x_shape = x.shape
            x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
            x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
            x = tfm.nlp.layers.KernelAttention(num_heads=num_heads[i],
                                               key_dim=key_dim[i],
                                               feature_transform="elu")(x, x)
            x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

            x_skip = tf.keras.layers.Conv2D(filters=x_skip.shape[-1],
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="same",
                                            kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)
            x_skip = tf.keras.layers.Add()([x_res, x_skip])

            x_res = x_skip

            for j in range(conv_layers[i]):
                x_skip = tf.keras.layers.Conv2D(filters=filters[i],
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="same",
                                                kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)

                x_conditioning_beta = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                             kernel_initializer=tf.keras.initializers.orthogonal)
                                       (x_conditioning))
                x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
                x_conditioning_beta = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                             kernel_initializer=tf.keras.initializers.orthogonal)
                                       (x_conditioning_beta))

                x_conditioning_gamma = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                              kernel_initializer=tf.keras.initializers.orthogonal)
                                        (x_conditioning))
                x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
                x_conditioning_gamma = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                              kernel_initializer=tf.keras.initializers.orthogonal)
                                        (x_conditioning_gamma))

                x_skip = tf.keras.layers.Multiply()([x_skip, x_conditioning_gamma])
                x_skip = tf.keras.layers.Add()([x_skip, x_conditioning_beta])

                x_skip = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_skip)

            x_res = tf.keras.layers.Conv2D(filters=x_skip.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
            x_skip = tf.keras.layers.Add()([x_skip, x_res])

        x_skips.append(x_skip)

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

        x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads[-1] is not None:
        x_res = x

        x = Standardiser()(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = tfm.nlp.layers.KernelAttention(num_heads=num_heads[-1],
                                           key_dim=key_dim[-1],
                                           feature_transform="elu")(x, x)
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

            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning))
            x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning_beta))

            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning))
            x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning_gamma))

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

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

        x_skip = x_skips.pop(-1)

        x_skip = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)
        x = tf.keras.layers.Concatenate()([x, x_skip])

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning))
            x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning_beta))

            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning))
            x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning_gamma))

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

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
    x = tf.keras.layers.Add()([x, x_noise_input])

    model = tf.keras.Model(inputs=[x_noise_input, x_timestep_encoding_input, x_label_input],
                           outputs=[x])

    return model


def get_model_conv_alex_image_input(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_noise_input = tf.keras.Input(shape=image_input_shape)
    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_timestep_encoding_input = tf.keras.Input(shape=1)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])
    x_counterfactual_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_noise_input)

    x_timestep_encoding = (
        tf.keras.layers.Embedding(input_dim=number_of_timesteps,
                                  output_dim=filters[-1],
                                  embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
        (x_timestep_encoding_input))
    x_timestep_encoding = x_timestep_encoding[:, 0]

    x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                         output_dim=filters[-1],
                                         embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
               (x_label_input))
    x_label = x_label[:, 0]

    x_timestep_encoding_image_conditioning = (
        tf.keras.layers.Dense(units=filters[-1],
                              kernel_initializer=tf.keras.initializers.orthogonal)(x_timestep_encoding))

    x_label_image_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label))

    x_image_conditioning = tf.keras.layers.Concatenate()([x_timestep_encoding_image_conditioning,
                                                          x_label_image_conditioning])

    for i in range(conditional_dense_layers):
        x_image_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_image_conditioning))
        x_image_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image_conditioning)

    x_image = tf.keras.layers.Conv2D(filters=filters[0],
                                     kernel_size=(7, 7),
                                     strides=(1, 1),
                                     padding="same",
                                     kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x_image

        for j in range(conv_layers[i]):
            x_image = tf.keras.layers.Conv2D(filters=filters[i],
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_image)

            # x_image_conditioning_beta = (tf.keras.layers.Dense(units=x_image.shape[-1],
            #                                                    kernel_initializer=tf.keras.initializers.orthogonal)
            #                              (x_image_conditioning))
            # x_image_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image_conditioning_beta)
            # x_image_conditioning_beta = (tf.keras.layers.Dense(units=x_image.shape[-1],
            #                                                    kernel_initializer=tf.keras.initializers.orthogonal)
            #                              (x_image_conditioning_beta))

            # x_image_conditioning_gamma = (tf.keras.layers.Dense(units=x_image.shape[-1],
            #                                                     kernel_initializer=tf.keras.initializers.orthogonal)
            #                               (x_image_conditioning))
            # x_image_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image_conditioning_gamma)
            # x_image_conditioning_gamma = (tf.keras.layers.Dense(units=x_image.shape[-1],
            #                                                     kernel_initializer=tf.keras.initializers.orthogonal)
            #                               (x_image_conditioning_gamma))

            # x_image = tf.keras.layers.Multiply()([x_image, x_image_conditioning_gamma])
            # x_image = tf.keras.layers.Add()([x_image, x_image_conditioning_beta])

            x_image = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image)

        x_res = tf.keras.layers.Conv2D(filters=x_image.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x_image = tf.keras.layers.Add()([x_image, x_res])

        x_image = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding="same",
                                         kernel_initializer=tf.keras.initializers.orthogonal)(x_image)
        x_image = tf.keras.layers.Lambda(einops.rearrange,
                                         arguments={"pattern": "b (h1 h2) (w1 w2) c -> b h1 w1 (c h2 w2)",
                                                    "h2": 2,
                                                    "w2": 2})(x_image)

    x_res = x_image

    for i in range(conv_layers[-1]):
        x_image = tf.keras.layers.Conv2D(filters=filters[-1],
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding="same",
                                         kernel_initializer=tf.keras.initializers.orthogonal)(x_image)

        # x_image_conditioning_beta = (tf.keras.layers.Dense(units=x_image.shape[-1],
        #                                                    kernel_initializer=tf.keras.initializers.orthogonal)
        #                              (x_image_conditioning))
        # x_image_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image_conditioning_beta)
        # x_image_conditioning_beta = (tf.keras.layers.Dense(units=x_image.shape[-1],
        #                                                    kernel_initializer=tf.keras.initializers.orthogonal)
        #                              (x_image_conditioning_beta))

        # x_image_conditioning_gamma = (tf.keras.layers.Dense(units=x_image.shape[-1],
        #                                                     kernel_initializer=tf.keras.initializers.orthogonal)
        #                               (x_image_conditioning))
        # x_image_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image_conditioning_gamma)
        # x_image_conditioning_gamma = (tf.keras.layers.Dense(units=x_image.shape[-1],
        #                                                     kernel_initializer=tf.keras.initializers.orthogonal)
        #                               (x_image_conditioning_gamma))

        # x_image = tf.keras.layers.Multiply()([x_image, x_image_conditioning_gamma])
        # x_image = tf.keras.layers.Add()([x_image, x_image_conditioning_beta])

        x_image = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image)

        x_res = tf.keras.layers.Conv2D(filters=x_image.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x_image = tf.keras.layers.Add()([x_image, x_res])

        # if num_heads[-1] is not None:
        #     x_res = x_image

        #     x_image = Standardiser()(x_image)

        #     x_image_shape = x_image.shape
        #     x_image_shape_prod = tf.math.reduce_prod(x_image_shape[1:-1]).numpy()
        #     x_image = tf.keras.layers.Reshape(target_shape=(x_image_shape_prod, x_image_shape[-1]))(x_image)
        #     x_image = (tfm.nlp.layers.KernelAttention(num_heads=num_heads[-1],
        #                                               key_dim=key_dim[-1],
        #                                               kernel_initializer=tf.keras.initializers.orthogonal,
        #                                               feature_transform="elu")(x_image, x_image))
        #     x_image = tf.keras.layers.Reshape(target_shape=x_image_shape[1:])(x_image)

        #     x_image = tf.keras.layers.Conv2D(filters=x_image.shape[-1],
        #                                      kernel_size=(1, 1),
        #                                      strides=(1, 1),
        #                                      padding="same",
        #                                      kernel_initializer=tf.keras.initializers.orthogonal)(x_image)
        #     x_image = tf.keras.layers.Add()([x_res, x_image])

        #     x_res = x_image

        #     for j in range(conv_layers[i]):
        #         x_image = tf.keras.layers.Conv2D(filters=filters[i],
        #                                          kernel_size=(3, 3),
        #                                          strides=(1, 1),
        #                                          padding="same",
        #                                          kernel_initializer=tf.keras.initializers.orthogonal)(x_image)

                # x_image_conditioning_beta = (tf.keras.layers.Dense(units=x_image.shape[-1],
                #                                                    kernel_initializer=tf.keras.initializers.orthogonal)
                #                              (x_image_conditioning))
                # x_image_conditioning_beta = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                #                              (x_image_conditioning_beta))
                # x_image_conditioning_beta = (tf.keras.layers.Dense(units=x_image.shape[-1],
                #                                                    kernel_initializer=tf.keras.initializers.orthogonal)
                #                              (x_image_conditioning_beta))

                # x_image_conditioning_gamma = (tf.keras.layers.Dense(units=x_image.shape[-1],
                #                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                #                               (x_image_conditioning))
                # x_image_conditioning_gamma = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                #                               (x_image_conditioning_gamma))
                # x_image_conditioning_gamma = (tf.keras.layers.Dense(units=x_image.shape[-1],
                #                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                #                               (x_image_conditioning_gamma))

                # x_image = tf.keras.layers.Multiply()([x_image, x_image_conditioning_gamma])
                # x_image = tf.keras.layers.Add()([x_image, x_image_conditioning_beta])

        #         x_image = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_image)

        #     x_res = tf.keras.layers.Conv2D(filters=x_image.shape[-1],
        #                                    kernel_size=(1, 1),
        #                                    strides=(1, 1),
        #                                    padding="same",
        #                                    kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        #     x_image = tf.keras.layers.Add()([x_image, x_res])

    x_image = tf.keras.layers.GlobalAvgPool2D()(x_image)

    x_image_bottleneck_size = int(np.ceil(x_image.shape[-1] / image_input_bottleneck_units_divider))
    x_image = tf.keras.layers.Dense(units=x_image_bottleneck_size,
                                    kernel_initializer=tf.keras.initializers.orthogonal)(x_image)

    x_image = tf.keras.layers.Dense(units=filters[-1],
                                    kernel_initializer=tf.keras.initializers.orthogonal)(x_image)

    x_timestep_encoding_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                              kernel_initializer=tf.keras.initializers.orthogonal)
                                        (x_timestep_encoding))

    x_label_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_label))

    x_conditioning = tf.keras.layers.Concatenate()([x_image, x_timestep_encoding_conditioning, x_label_conditioning])

    for i in range(conditional_dense_layers):
        x_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                kernel_initializer=tf.keras.initializers.orthogonal)
                          (x_conditioning))
        x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    x_timestep_encoding_counterfactual_conditioning = (
        tf.keras.layers.Dense(units=filters[-1],
                              kernel_initializer=tf.keras.initializers.orthogonal)
        (x_timestep_encoding))

    x_counterfactual_label = (
        tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                  output_dim=filters[-1],
                                  embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
        (x_counterfactual_label_input))
    x_counterfactual_label = x_counterfactual_label[:, 0]

    x_counterfactual_conditioning = tf.keras.layers.Concatenate()([x_timestep_encoding_counterfactual_conditioning,
                                                                   x_counterfactual_label])

    for i in range(conditional_dense_layers):
        x_counterfactual_conditioning = (tf.keras.layers.Dense(units=filters[-1],
                                                               kernel_initializer=tf.keras.initializers.orthogonal)
                                         (x_counterfactual_conditioning))
        x_counterfactual_conditioning = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                         (x_counterfactual_conditioning))

    x_skips = []

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning))
            x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
            x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                         kernel_initializer=tf.keras.initializers.orthogonal)
                                   (x_conditioning_beta))

            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning))
            x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
            x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                          kernel_initializer=tf.keras.initializers.orthogonal)
                                    (x_conditioning_gamma))

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x_skip = x

        if num_heads[i] is not None:
            x_res = x_skip

            for j in range(conv_layers[i]):
                x_skip = tf.keras.layers.Conv2D(filters=filters[i],
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="same",
                                                kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)

                x_conditioning_beta = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                             kernel_initializer=tf.keras.initializers.orthogonal)
                                       (x_conditioning))
                x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
                x_conditioning_beta = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                             kernel_initializer=tf.keras.initializers.orthogonal)
                                       (x_conditioning_beta))

                x_conditioning_gamma = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                              kernel_initializer=tf.keras.initializers.orthogonal)
                                        (x_conditioning))
                x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
                x_conditioning_gamma = (tf.keras.layers.Dense(units=x_skip.shape[-1],
                                                              kernel_initializer=tf.keras.initializers.orthogonal)
                                        (x_conditioning_gamma))

                x_skip = tf.keras.layers.Multiply()([x_skip, x_conditioning_gamma])
                x_skip = tf.keras.layers.Add()([x_skip, x_conditioning_beta])

                x_skip = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_skip)

            x_res = tf.keras.layers.Conv2D(filters=x_skip.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
            x_skip = tf.keras.layers.Add()([x_skip, x_res])

            x_res = x_skip

            x_skip = Standardiser()(x_skip)

            x_skip_shape = x_skip.shape
            x_skip_shape_prod = tf.math.reduce_prod(x_skip_shape[1:-1]).numpy()
            x_skip = tf.keras.layers.Reshape(target_shape=(x_skip_shape_prod, x_skip_shape[-1]))(x_skip)
            x_skip = (tfm.nlp.layers.KernelAttention(num_heads=num_heads[i],
                                                     key_dim=key_dim[i],
                                                     kernel_initializer=tf.keras.initializers.orthogonal,
                                                     feature_transform="elu")(x_skip, x_skip))
            x_skip = tf.keras.layers.Reshape(target_shape=x_skip_shape[1:])(x_skip)

            x_skip = tf.keras.layers.Conv2D(filters=x_skip.shape[-1],
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="same",
                                            kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)
            x_skip = tf.keras.layers.Add()([x_res, x_skip])

            x_res = x_skip

            for j in range(conv_layers[i]):
                x_skip = tf.keras.layers.Conv2D(filters=filters[i],
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="same",
                                                kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)

                x_counterfactual_conditioning_beta = (
                    tf.keras.layers.Dense(units=x.shape[-1],
                                          kernel_initializer=tf.keras.initializers.orthogonal)
                    (x_counterfactual_conditioning))
                x_counterfactual_conditioning_beta = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                                      (x_counterfactual_conditioning_beta))
                x_counterfactual_conditioning_beta = (
                    tf.keras.layers.Dense(units=x.shape[-1],
                                          kernel_initializer=tf.keras.initializers.orthogonal)
                    (x_counterfactual_conditioning_beta))

                x_counterfactual_conditioning_gamma = (
                    tf.keras.layers.Dense(units=x.shape[-1],
                                          kernel_initializer=tf.keras.initializers.orthogonal)(
                        x_counterfactual_conditioning))
                x_counterfactual_conditioning_gamma = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                                       (x_counterfactual_conditioning_gamma))
                x_counterfactual_conditioning_gamma = (
                    tf.keras.layers.Dense(units=x.shape[-1],
                                          kernel_initializer=tf.keras.initializers.orthogonal)
                    (x_counterfactual_conditioning_gamma))

                x_skip = tf.keras.layers.Multiply()([x_skip, x_counterfactual_conditioning_gamma])
                x_skip = tf.keras.layers.Add()([x_skip, x_counterfactual_conditioning_beta])

                x_skip = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_skip)

            x_res = tf.keras.layers.Conv2D(filters=x_skip.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
            x_skip = tf.keras.layers.Add()([x_skip, x_res])

        x_skips.append(x_skip)

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

        x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads[-1] is not None:
        x_res = x

        x = Standardiser()(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads[-1],
                                            key_dim=key_dim[-1],
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

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)

        x_counterfactual_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                                    kernel_initializer=tf.keras.initializers.orthogonal)
                                              (x_counterfactual_conditioning))
        x_counterfactual_conditioning_beta = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                              (x_counterfactual_conditioning_beta))
        x_counterfactual_conditioning_beta = (tf.keras.layers.Dense(units=x.shape[-1],
                                                                    kernel_initializer=tf.keras.initializers.orthogonal)
                                              (x_counterfactual_conditioning_beta))

        x_counterfactual_conditioning_gamma = (
            tf.keras.layers.Dense(units=x.shape[-1],
                                  kernel_initializer=tf.keras.initializers.orthogonal)(x_counterfactual_conditioning))
        x_counterfactual_conditioning_gamma = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                               (x_counterfactual_conditioning_gamma))
        x_counterfactual_conditioning_gamma = (
            tf.keras.layers.Dense(units=x.shape[-1],
                                  kernel_initializer=tf.keras.initializers.orthogonal)
            (x_counterfactual_conditioning_gamma))

        x = tf.keras.layers.Multiply()([x, x_counterfactual_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_counterfactual_conditioning_beta])

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

        x_skip = x_skips.pop(-1)

        x_skip = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_skip)
        x = tf.keras.layers.Concatenate()([x, x_skip])

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)

            x_counterfactual_conditioning_beta = (
                tf.keras.layers.Dense(units=x.shape[-1],
                                      kernel_initializer=tf.keras.initializers.orthogonal)
                (x_counterfactual_conditioning))
            x_counterfactual_conditioning_beta = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                                  (x_counterfactual_conditioning_beta))
            x_counterfactual_conditioning_beta = (
                tf.keras.layers.Dense(units=x.shape[-1],
                                      kernel_initializer=tf.keras.initializers.orthogonal)
                (x_counterfactual_conditioning_beta))

            x_counterfactual_conditioning_gamma = (
                tf.keras.layers.Dense(units=x.shape[-1],
                                      kernel_initializer=tf.keras.initializers.orthogonal)
                (x_counterfactual_conditioning))
            x_counterfactual_conditioning_gamma = (tf.keras.layers.Lambda(tf.keras.activations.swish)
                                                   (x_counterfactual_conditioning_gamma))
            x_counterfactual_conditioning_gamma = (
                tf.keras.layers.Dense(units=x.shape[-1],
                                      kernel_initializer=tf.keras.initializers.orthogonal)
                (x_counterfactual_conditioning_gamma))

            x = tf.keras.layers.Multiply()([x, x_counterfactual_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_counterfactual_conditioning_beta])

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
    x = tf.keras.layers.Add()([x, x_noise_input])

    model = tf.keras.Model(inputs=[x_noise_input, x_image_input, x_timestep_encoding_input, x_label_input,
                                   x_counterfactual_label_input], outputs=[x])

    return model


def sinspace(start, stop, num):
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

    batch_sizes_epochs = sinspace(0.0, epochs - 1, len(batch_sizes) + 1)
    batch_sizes_epochs = np.round(batch_sizes_epochs)

    return batch_sizes, batch_sizes_epochs


def cosspace(start, stop, num):
    linspaced = np.linspace(-180.0, -90.0, num)

    cosspaced = np.cos(np.deg2rad(linspaced))

    cosspaced_min = np.min(cosspaced)
    cosspaced = start + ((cosspaced - cosspaced_min) * ((stop - start) / ((np.max(cosspaced) - cosspaced_min) +
                                                                          np.finfo(np.float32).eps)))

    return cosspaced


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


# https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac
# this function will add noise to the input as per the given timestamp
def forward_noise(x_zero, t, sqrt_alpha_bar, one_minus_sqrt_alpha_bar):
    noise = tf.random.normal(x_zero.shape)

    noisy_image = ((np.reshape(np.take(sqrt_alpha_bar, t), (-1, 1, 1, 1)) * x_zero) +
                   (np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1)) * noise))

    return noisy_image, noise


def mean_squared_error(y_true, y_pred):
    loss = tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2.0))

    return loss


def root_mean_squared_error(y_true, y_pred):
    loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2.0)) + tf.keras.backend.epsilon())

    return loss


def get_error(y_true, y_pred):
    error = tf.math.reduce_mean(tf.math.abs(tf.math.log(tf.math.abs(tf.math.divide_no_nan(y_pred, y_true)) +
                                                        tf.keras.backend.epsilon()))) * 100.0

    return error


# https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac
def ddpm(x_t, y_pred, t, beta, alpha, alpha_bar):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    mean = (1.0 / (np.power(alpha_t, 0.5))) * (x_t - (((1.0 - alpha_t) / np.power(1.0 - alpha_t_bar, 0.5)) * y_pred))
    var = np.take(beta, t)
    z = tf.random.normal(x_t.shape)

    x_t_minus_one = mean + (np.power(var, 0.5) * z)

    return x_t_minus_one


def output_image(image, standard_scaler, current_output_path):
    image = image.numpy()

    image = np.reshape(standard_scaler.inverse_transform(np.reshape(image, (-1, 1))), image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    if greyscale_bool:
        image = image[:, :, 0]

    image = Image.fromarray(image)
    image.save(current_output_path)

    return True


def validate(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images, standard_scaler,
             x_timestep_encodings, x_test_labels, i):
    validate_output_path = "{0}/validate/".format(output_path)

    output_input_bool = False

    if not os.path.exists(validate_output_path):
        mkdir_p(validate_output_path)

        output_input_bool = True

    current_validate_output_path = "{0}/{1}/".format(validate_output_path, str(i))
    mkdir_p(current_validate_output_path)

    current_x_test_image = get_data_from_storage(x_test_images[0])

    current_x_test_label = x_test_labels[0]
    counterfactual_x_test_label = x_test_labels[8]

    current_x_test_image = tf.expand_dims(current_x_test_image, axis=0)
    current_x_test_label = tf.expand_dims(current_x_test_label, axis=0)
    counterfactual_x_test_label = tf.expand_dims(counterfactual_x_test_label, axis=0)

    if output_input_bool:
        output_image(current_x_test_image[0], standard_scaler, "{0}/input.png".format(validate_output_path))

    start_timestep = int(np.round(number_of_timesteps * output_start_timestep_proportion))

    current_x_test_image_with_noise, noise = (
        forward_noise(current_x_test_image, start_timestep - 1, sqrt_alpha_bar, one_minus_sqrt_alpha_bar))

    output_image(current_x_test_image_with_noise[0], standard_scaler, "{0}/{1}.png".format(
        current_validate_output_path, str(start_timestep)))

    for j in range(start_timestep - 1, 0, -1):
        if alex_bool:
            current_x_timestep_encoding = tf.convert_to_tensor(j)
            current_x_timestep_encoding = tf.expand_dims(current_x_timestep_encoding, axis=0)

            if image_input_bool:
                y_pred = model([current_x_test_image_with_noise, current_x_test_image, current_x_timestep_encoding,
                                current_x_test_label, counterfactual_x_test_label], training=False)
            else:
                y_pred = model([current_x_test_image_with_noise, current_x_timestep_encoding,
                                counterfactual_x_test_label], training=False)
        else:
            current_x_timestep_encoding = x_timestep_encodings[j]
            current_x_timestep_encoding = tf.expand_dims(current_x_timestep_encoding, axis=0)

            y_pred = model([current_x_test_image_with_noise, current_x_timestep_encoding,
                            counterfactual_x_test_label], training=False)

        current_x_test_image_with_noise = ddpm(current_x_test_image_with_noise, y_pred, j, beta, alpha, alpha_bar)

        output_image(current_x_test_image_with_noise[0], standard_scaler, "{0}/{1}.png".format(
            current_validate_output_path, str(j)))

    output_image(current_x_test_image_with_noise[0], standard_scaler, "{0}/{1}.png".format(
        validate_output_path, str(i)))

    return True


def train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, beta, alpha, alpha_bar,
                                sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_train_images, x_test_images,
                                standard_scaler, x_timestep_encodings, x_train_labels, x_test_labels):
    print("train")

    validate(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images,
             standard_scaler, x_timestep_encodings, x_test_labels, 0)

    x_train_images_len = len(x_train_images)

    current_batch_size = None
    indices = list(range(x_train_images_len))
    timesteps = []

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

            for m in range(current_batch_size):
                current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])

                current_x_train_image = augmentation(current_x_train_image)
                current_x_train_label = x_train_labels[indices[current_index]]

                current_x_train_image = tf.expand_dims(current_x_train_image, axis=0)
                current_x_train_label = tf.expand_dims(current_x_train_label, axis=0)

                if len(timesteps) < 1:
                    timesteps = list(range(number_of_timesteps))
                    random.shuffle(timesteps)

                current_timestep = timesteps.pop(0)

                if alex_bool:
                    current_x_timestep_encoding = tf.convert_to_tensor(current_timestep)
                    current_x_timestep_encoding = tf.expand_dims(current_x_timestep_encoding, axis=0)
                else:
                    current_x_timestep_encoding = x_timestep_encodings[current_timestep]
                    current_x_timestep_encoding = tf.expand_dims(current_x_timestep_encoding, axis=0)

                current_x_train_image_with_noise, y_true = (
                    forward_noise(current_x_train_image, current_timestep, sqrt_alpha_bar, one_minus_sqrt_alpha_bar))

                if alex_bool:
                    if image_input_bool:
                        if i + 1 > mean_squared_error_epochs:
                            with tf.GradientTape() as tape:
                                y_pred = model([current_x_train_image_with_noise, current_x_train_image,
                                                current_x_timestep_encoding, current_x_train_label,
                                                current_x_train_label], training=True)

                                loss = tf.math.reduce_sum([root_mean_squared_error(y_true, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred = model([current_x_train_image_with_noise, current_x_train_image,
                                                current_x_timestep_encoding, current_x_train_label,
                                                current_x_train_label], training=True)

                                loss = tf.math.reduce_sum([mean_squared_error(y_true, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                    else:
                        if i + 1 > mean_squared_error_epochs:
                            with tf.GradientTape() as tape:
                                y_pred = model([current_x_train_image_with_noise, current_x_timestep_encoding,
                                                current_x_train_label], training=True)

                                loss = tf.math.reduce_sum([root_mean_squared_error(y_true, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred = model([current_x_train_image_with_noise, current_x_timestep_encoding,
                                                current_x_train_label], training=True)

                                loss = tf.math.reduce_sum([mean_squared_error(y_true, y_pred),
                                                           tf.math.reduce_sum(model.losses)])
                else:
                    if i + 1 > mean_squared_error_epochs:
                        with tf.GradientTape() as tape:
                            y_pred = model([current_x_train_image_with_noise, current_x_timestep_encoding,
                                            current_x_train_label], training=True)

                            loss = tf.math.reduce_sum([root_mean_squared_error(y_true, y_pred),
                                                       tf.math.reduce_sum(model.losses)])
                    else:
                        with tf.GradientTape() as tape:
                            y_pred = model([current_x_train_image_with_noise, current_x_timestep_encoding,
                                            current_x_train_label], training=True)

                            loss = tf.math.reduce_sum([mean_squared_error(y_true, y_pred),
                                                       tf.math.reduce_sum(model.losses)])

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)
                errors.append(get_error(y_true, y_pred))

                current_index = current_index + 1

            gradients = [gradient / current_batch_size for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            error = tf.math.reduce_mean(errors)

            if i + 1 > mean_squared_error_epochs:
                loss_name = "RMSE"
            else:
                loss_name = "MSE"

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss ({5}): {6:14} Error: {7:14}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), loss_name,
                str(loss.numpy()), str(error.numpy())))

        validate(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images,
                 standard_scaler, x_timestep_encodings, x_test_labels, i + 1)

    return model


def train(model, optimiser, batch_sizes, batch_sizes_epochs, beta, alpha, alpha_bar, sqrt_alpha_bar,
          one_minus_sqrt_alpha_bar, x_train_images, x_test_images, standard_scaler, x_timestep_encodings,
          x_train_labels, x_test_labels):
    print("train")

    validate(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images,
             standard_scaler, x_timestep_encodings, x_test_labels, 0)

    x_train_images_len = len(x_train_images)

    current_batch_size = None
    indices = list(range(x_train_images_len))
    timesteps = []

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
            current_x_train_labels = []

            for k in range(current_batch_size):
                current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])

                current_x_train_images.append(augmentation(current_x_train_image))
                current_x_train_labels.append(x_train_labels[indices[current_index]])

                current_index = current_index + 1

            current_x_train_images = tf.convert_to_tensor(current_x_train_images)
            current_x_train_labels = tf.convert_to_tensor(current_x_train_labels)

            current_x_timestep_encodings = []

            if unbatch_timesteps_bool:
                current_x_train_images_with_noise = []
                y_true = []

                for k in range(current_batch_size):
                    if len(timesteps) < 1:
                        timesteps = list(range(number_of_timesteps))
                        random.shuffle(timesteps)

                    current_timestep = timesteps.pop(0)

                    if alex_bool:
                        current_x_timestep_encodings.append(current_timestep)
                    else:
                        current_x_timestep_encodings.append(x_timestep_encodings[current_timestep])

                    current_x_train_image_with_noise, current_noise = (
                        forward_noise(tf.expand_dims(current_x_train_images[k], axis=0), current_timestep,
                                      sqrt_alpha_bar, one_minus_sqrt_alpha_bar))

                    current_x_train_images_with_noise.append(current_x_train_image_with_noise[0])
                    y_true.append(current_noise[0])

                current_x_timestep_encodings = tf.convert_to_tensor(current_x_timestep_encodings)
                current_x_train_images_with_noise = tf.convert_to_tensor(current_x_train_images_with_noise)
                y_true = tf.convert_to_tensor(y_true)
            else:
                if len(timesteps) < 1:
                    timesteps = list(range(number_of_timesteps))
                    random.shuffle(timesteps)

                current_timestep = timesteps.pop(0)

                if alex_bool:
                    for k in range(current_batch_size):
                        current_x_timestep_encodings.append(current_timestep)
                else:
                    for k in range(current_batch_size):
                        current_x_timestep_encodings.append(x_timestep_encodings[current_timestep])

                current_x_timestep_encodings = tf.convert_to_tensor(current_x_timestep_encodings)

                current_x_train_images_with_noise, y_true = (
                    forward_noise(current_x_train_images, current_timestep, sqrt_alpha_bar, one_minus_sqrt_alpha_bar))

            if alex_bool:
                if image_input_bool:
                    if i + 1 > mean_squared_error_epochs:
                        with tf.GradientTape() as tape:
                            y_pred = model([current_x_train_images_with_noise, current_x_train_images,
                                            current_x_timestep_encodings, current_x_train_labels,
                                            current_x_train_labels], training=True)

                            loss = tf.math.reduce_sum([root_mean_squared_error(y_true, y_pred),
                                                       tf.math.reduce_sum(model.losses)])
                    else:
                        with tf.GradientTape() as tape:
                            y_pred = model([current_x_train_images_with_noise, current_x_train_images,
                                            current_x_timestep_encodings, current_x_train_labels,
                                            current_x_train_labels], training=True)

                            loss = tf.math.reduce_sum([mean_squared_error(y_true, y_pred),
                                                       tf.math.reduce_sum(model.losses)])
                else:
                    if i + 1 > mean_squared_error_epochs:
                        with tf.GradientTape() as tape:
                            y_pred = model([current_x_train_images_with_noise, current_x_timestep_encodings,
                                            current_x_train_labels], training=True)

                            loss = tf.math.reduce_sum([root_mean_squared_error(y_true, y_pred),
                                                       tf.math.reduce_sum(model.losses)])
                    else:
                        with tf.GradientTape() as tape:
                            y_pred = model([current_x_train_images_with_noise, current_x_timestep_encodings,
                                            current_x_train_labels], training=True)

                            loss = tf.math.reduce_sum([mean_squared_error(y_true, y_pred),
                                                       tf.math.reduce_sum(model.losses)])
            else:
                if i + 1 > mean_squared_error_epochs:
                    with tf.GradientTape() as tape:
                        y_pred = model([current_x_train_images_with_noise, current_x_timestep_encodings,
                                        current_x_train_labels], training=True)

                        loss = tf.math.reduce_sum([root_mean_squared_error(y_true, y_pred),
                                                   tf.math.reduce_sum(model.losses)])
                else:
                    with tf.GradientTape() as tape:
                        y_pred = model([current_x_train_images_with_noise, current_x_timestep_encodings,
                                        current_x_train_labels], training=True)

                        loss = tf.math.reduce_sum([mean_squared_error(y_true, y_pred),
                                                   tf.math.reduce_sum(model.losses)])

            gradients = tape.gradient(loss, model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            error = get_error(y_true, y_pred)

            if i + 1 > mean_squared_error_epochs:
                loss_name = "RMSE"
            else:
                loss_name = "MSE"

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss ({5}): {6:14} Error: {7:14}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), loss_name,
                str(loss.numpy()), str(error.numpy())))

        validate(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images,
                 standard_scaler, x_timestep_encodings, x_test_labels, i + 1)

    return model


def test(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images, standard_scaler,
         x_timestep_encodings, x_test_labels):
    print("test")

    test_output_path = "{0}/test/".format(output_path)
    mkdir_p(test_output_path)

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

        for j in range(test_batch_size):
            current_index = index + j

            output_image(current_x_test_images[j], standard_scaler,
                         "{0}/{1}_input.png".format(test_output_path, str(current_index)))

        start_timestep = int(np.round(number_of_timesteps * output_start_timestep_proportion))

        current_x_test_images_with_noise, noise = (
            forward_noise(current_x_test_images, start_timestep - 1, sqrt_alpha_bar, one_minus_sqrt_alpha_bar))

        for j in range(test_batch_size):
            current_index = index + j

            current_index_test_output_path = "{0}/{1}/".format(test_output_path, str(current_index))
            mkdir_p(current_index_test_output_path)

        for k in range(test_batch_size):
            current_index = index + k

            current_index_test_output_path = "{0}/{1}/".format(test_output_path, str(current_index))

            output_image(current_x_test_images_with_noise[k], standard_scaler, "{0}/{1}.png".format(
                current_index_test_output_path, str(start_timestep)))

        for j in range(start_timestep - 1, 0, -1):
            current_x_timestep_encodings = []

            if alex_bool:
                for k in range(test_batch_size):
                    current_x_timestep_encodings.append(j)
            else:
                for k in range(test_batch_size):
                    current_x_timestep_encodings.append(x_timestep_encodings[j])

            current_x_timestep_encodings = tf.convert_to_tensor(current_x_timestep_encodings)

            if alex_bool:
                if image_input_bool:
                    y_pred = model([current_x_test_images_with_noise, current_x_test_images,
                                    current_x_timestep_encodings, current_x_test_labels, current_x_test_labels],
                                   training=False)
                else:
                    y_pred = model([current_x_test_images_with_noise, current_x_timestep_encodings,
                                    current_x_test_labels], training=False)
            else:
                y_pred = model([current_x_test_images_with_noise, current_x_timestep_encodings,
                                current_x_test_labels], training=False)

            current_x_test_images_with_noise = ddpm(current_x_test_images_with_noise, y_pred, j, beta, alpha, alpha_bar)

            for k in range(test_batch_size):
                current_index = index + k

                current_index_test_output_path = "{0}/{1}/".format(test_output_path, str(current_index))

                output_image(current_x_test_images_with_noise[k], standard_scaler, "{0}/{1}.png".format(
                    current_index_test_output_path, str(j)))

        for j in range(test_batch_size):
            current_index = index + j

            output_image(current_x_test_images_with_noise[j], standard_scaler, "{0}/{1}.png".format(
                test_output_path, str(current_index)))

    return True


def main():
    print("main")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if read_data_from_storage_bool:
        mkdir_p(output_path)

    x_train_images, x_test_images, x_train_labels, x_test_labels = get_input()
    x_timestep_encodings = get_positional_encodings(filters[-1])
    x_train_images, x_test_images, standard_scaler, x_timestep_encodings, x_train_labels, x_test_labels = (
        preprocess_input(x_train_images, x_test_images, x_timestep_encodings, x_train_labels, x_test_labels))

    if alex_bool:
        if image_input_bool:
            model = get_model_conv_alex_image_input(x_train_images, x_train_labels)
        else:
            model = get_model_conv_alex(x_train_images, x_train_labels)
    else:
        model = get_model_conv(x_train_images, x_timestep_encodings, x_train_labels)

    model.summary()

    if alex_bool:
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             weight_decay=weight_decay,
                                             use_ema=True,
                                             ema_overwrite_frequency=ema_overwrite_frequency)
    else:
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.99,
                                             use_ema=True,
                                             ema_momentum=0.995,
                                             ema_overwrite_frequency=ema_overwrite_frequency)

    batch_sizes, batch_sizes_epochs = get_batch_sizes(x_train_images)

    beta = cosspace(1e-04, 0.02, number_of_timesteps)

    alpha = 1 - beta
    alpha_bar = np.concatenate((np.array([1.0]), np.cumprod(alpha, axis=0)[:-1]), axis=0)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    one_minus_sqrt_alpha_bar = np.sqrt(1 - alpha_bar)

    if gradient_accumulation_bool:
        model = train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, beta, alpha, alpha_bar,
                                            sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_train_images, x_test_images,
                                            standard_scaler, x_timestep_encodings, x_train_labels, x_test_labels)
    else:
        model = train(model, optimiser, batch_sizes, batch_sizes_epochs, beta, alpha, alpha_bar, sqrt_alpha_bar,
                      one_minus_sqrt_alpha_bar, x_train_images, x_test_images, standard_scaler, x_timestep_encodings,
                      x_train_labels, x_test_labels)

    if epochs > 0:
        optimiser.finalize_variable_values(model.trainable_variables)

    test(model, beta, alpha, alpha_bar, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, x_test_images, standard_scaler,
         x_timestep_encodings, x_test_labels)

    return True


if __name__ == "__main__":
    main()
