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
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import einops
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm
from vector_quantiser import VectorQuantiser
import pickle
from PIL import Image

np.seterr(all="print")

dataset_name = "cifar10"
output_path = "../output/autoencoder/"

read_data_from_storage_bool = False

preprocess_list_bool = False
greyscale_bool = True
output_dimension_size = 32

deep_bool = True
conv_bool = True
alex_bool = True
gaussian_latent_bool = True
discrete_bool = False
gaussian_negative_log_likelihood_bool = False

dense_layers = 4
conditioning_input_size_divisor = 1.0
conditioning_input_embedding_bools = [True]
number_of_label_bins = 256
conditional_dense_layers = 2
filters = [32, 64, 128, 256, 512, 1024]

if gaussian_latent_bool:
    latent_filters = 1024
else:
    if discrete_bool:
        latent_filters = 1024
    else:
        latent_filters = 1024

conv_layers = [2, 2, 2, 2, 2, 2]
num_heads = 4
key_dim = 32
num_embeddings = 16384

conditioning_input_size = int(np.round(filters[-1] / conditioning_input_size_divisor))
conditioning_inputs_size = [int(np.round(conditioning_input_size / len(conditioning_input_embedding_bools))),
                            conditioning_input_size]

if alex_bool:
    learning_rate = 1e-04
    weight_decay = 0.0
    use_ema = False
    ema_overwrite_frequency = None
else:
    learning_rate = 1e-04
    weight_decay = 0.0
    use_ema = False
    ema_overwrite_frequency = None

gradient_accumulation_bool = False

epochs = 256

if gradient_accumulation_bool:
    gradient_accumulation_batch_size = 1

    min_batch_size = 32
    max_batch_size = 32
else:
    if alex_bool:
        min_batch_size = 32
        max_batch_size = 32
    else:
        min_batch_size = 32
        max_batch_size = 32

augmentation_bool = False
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
    if gaussian_latent_bool:
        kullback_leibler_divergence_epochs = 0
        discrete_mean_squared_error_epochs = 0
        mean_squared_error_epochs = 1
    else:
        if discrete_bool:
            kullback_leibler_divergence_epochs = 0
            discrete_mean_squared_error_epochs = 0
            mean_squared_error_epochs = 1
        else:
            kullback_leibler_divergence_epochs = 0
            discrete_mean_squared_error_epochs = 0
            mean_squared_error_epochs = 1
else:
    kullback_leibler_divergence_epochs = 0
    discrete_mean_squared_error_epochs = 0
    mean_squared_error_epochs = epochs

gaussian_latent_loss_weight = 1e-01


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
    # dataset_test = dataset["validation"]

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
            data_from_storage = pickle.load(file)
    else:
        data_from_storage = copy.deepcopy(data)

    return data_from_storage


def set_data_from_storage(data, current_output_path):
    if read_data_from_storage_bool:
        with open(current_output_path, "wb") as file:
            pickle.dump(data, file)

        data_from_storage = current_output_path
    else:
        data_from_storage = copy.deepcopy(data)

    return data_from_storage


def convert_rgb_to_greyscale(images):
    print("convert_rgb_to_greyscale")

    greyscale_images = []

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        if image.shape[-1] > 1:
            image = rgb2gray(image)
            image = np.expand_dims(image, axis=-1)

        greyscale_images.append(set_data_from_storage(image, images[i]))

    return greyscale_images


def get_next_geometric_value(value, base):
    power = np.log2(value / base) + 1

    if not power.is_integer():
        next_value = base * np.power(2.0, (np.ceil(power) - 1.0))
    else:
        next_value = copy.deepcopy(value)

    return next_value


def rescale_images_list(images, rescaled_dimension_size):
    print("rescale_images")

    rescaled_images = []

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        image = rescale(image, rescaled_dimension_size / np.max(image.shape[:-1]), order=3, preserve_range=True,
                        channel_axis=-1)

        rescaled_images.append(set_data_from_storage(image, images[i]))

    return rescaled_images


def normalise_images_list(x_train_images, x_test_images):
    print("normalise_images")

    standard_scaler = StandardScaler()

    x_train_images_len = len(x_train_images)

    for i in range(x_train_images_len):
        current_x_train_images = get_data_from_storage(x_train_images[i])

        standard_scaler.partial_fit(np.reshape(current_x_train_images, (-1, 1)))

    normalised_x_train_images = []

    for i in range(x_train_images_len):
        current_x_train_images = get_data_from_storage(x_train_images[i])

        current_x_train_images = np.reshape(standard_scaler.transform(np.reshape(current_x_train_images,
                                                                                 (-1, 1))),
                                            current_x_train_images.shape)

        normalised_x_train_images.append(set_data_from_storage(current_x_train_images, x_train_images[i]))

    normalised_x_test_images = []

    for i in range(len(x_test_images)):
        current_x_test_images = get_data_from_storage(x_test_images[i])

        current_x_test_images = np.reshape(standard_scaler.transform(np.reshape(current_x_test_images,
                                                                                (-1, 1))),
                                           current_x_test_images.shape)

        normalised_x_test_images.append(set_data_from_storage(current_x_test_images, x_test_images[i]))

    return normalised_x_train_images, normalised_x_test_images, standard_scaler


def pad_image(image, padded_dimension_size, output_shape=None):
    padded_image = copy.deepcopy(image)

    if output_shape is not None:
        padded_dimension_size = output_shape[0]

    while padded_image.shape[0] + 1 < padded_dimension_size:
        padded_image = np.pad(padded_image, ((1, 1), (0, 0), (0, 0)))

    if padded_image.shape[0] < padded_dimension_size:
        padded_image = np.pad(padded_image, ((0, 1), (0, 0), (0, 0)))

    if output_shape is not None:
        padded_dimension_size = output_shape[1]

    while padded_image.shape[1] + 1 < padded_dimension_size:
        padded_image = np.pad(padded_image, ((0, 0), (1, 1), (0, 0)))

    if padded_image.shape[1] < padded_dimension_size:
        padded_image = np.pad(padded_image, ((0, 0), (0, 1), (0, 0)))

    return padded_image


def pad_images_list(images, padded_dimension_size, current_output_path):
    print("pad_images")

    padded_images = []
    original_shapes = []
    padding_masks = []

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        padding_mask = np.ones(image.shape, dtype=np.float32)

        original_shapes.append(image.shape)
        image = pad_image(image, padded_dimension_size)

        padding_mask = pad_image(padding_mask, padded_dimension_size)

        padded_images.append(set_data_from_storage(image, images[i]))

        if read_data_from_storage_bool:
            images_split = images[i].strip().split('/')[-1].split('.')
            padding_masks.append("{0}/{1}_padding_mask.{2}".format(current_output_path, images_split[0],
                                                                   images_split[1]))

        padding_masks[i] = set_data_from_storage(padding_mask, padding_masks[i])

    return padded_images, original_shapes, padding_masks


def convert_to_boolean_mask_list(images):
    print("convert_to_boolean_mask_list")

    boolean_masks = []

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        image = image.astype(bool)

        boolean_masks.append(set_data_from_storage(image, images[i]))

    return boolean_masks


def convert_images_to_tensor_list(images):
    print("convert_images_to_tensor")

    tensors = []

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        image = tf.convert_to_tensor(image)

        tensors.append(set_data_from_storage(image, images[i]))

    return tensors


def preprocess_images_list(x_train_images, x_test_images):
    print("preprocess_images")

    x_train_images_preprocessed = rescale_images_list(x_train_images, output_dimension_size)
    x_test_images_preprocessed = rescale_images_list(x_test_images, output_dimension_size)

    x_train_images_preprocessed, x_test_images_preprocessed, standard_scaler = (
        normalise_images_list(x_train_images_preprocessed, x_test_images_preprocessed))

    x_train_padding_masks_output_path = "{0}/x_train_padding_masks/".format(output_path)

    if read_data_from_storage_bool:
        mkdir_p(x_train_padding_masks_output_path)

    x_test_padding_masks_output_path = "{0}/x_test_padding_masks/".format(output_path)

    if read_data_from_storage_bool:
        mkdir_p(x_test_padding_masks_output_path)

    x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks = (
        pad_images_list(x_train_images_preprocessed, output_dimension_size, x_train_padding_masks_output_path))
    x_test_images_preprocessed, x_test_original_shapes, x_test_padding_masks = (
        pad_images_list(x_test_images_preprocessed, output_dimension_size, x_test_padding_masks_output_path))

    if read_data_from_storage_bool:
        x_train_images_preprocessed = convert_images_to_tensor_list(x_train_images_preprocessed)
        x_test_images_preprocessed = convert_images_to_tensor_list(x_test_images_preprocessed)

        x_train_padding_masks = convert_to_boolean_mask_list(x_train_padding_masks)
        x_test_padding_masks = convert_to_boolean_mask_list(x_test_padding_masks)

        x_train_padding_masks = convert_images_to_tensor_list(x_train_padding_masks)
        x_test_padding_masks = convert_images_to_tensor_list(x_test_padding_masks)
    else:
        x_train_images_preprocessed = np.array(x_train_images_preprocessed)
        x_test_images_preprocessed = np.array(x_test_images_preprocessed)

        x_train_padding_masks = np.array(x_train_padding_masks)
        x_test_padding_masks = np.array(x_test_padding_masks)

        x_train_images_preprocessed = tf.convert_to_tensor(x_train_images_preprocessed)
        x_test_images_preprocessed = tf.convert_to_tensor(x_test_images_preprocessed)

        x_train_padding_masks = x_train_padding_masks.astype(bool)
        x_test_padding_masks = x_test_padding_masks.astype(bool)

        x_train_padding_masks = tf.convert_to_tensor(x_train_padding_masks)
        x_test_padding_masks = tf.convert_to_tensor(x_test_padding_masks)

    return (x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks, x_test_images_preprocessed,
            x_test_original_shapes, x_test_padding_masks, standard_scaler)


def rescale_images_array(images, rescaled_dimension_size):
    print("rescale_images")

    rescaled_images = []

    for i in range(len(images)):
        rescaled_images.append(rescale(images[i], rescaled_dimension_size / np.max(images[i].shape[:-1]), order=3,
                                       preserve_range=True, channel_axis=-1))

    rescaled_images = np.array(rescaled_images)

    return rescaled_images


def normalise_images_array(x_train_images, x_test_images):
    print("normalise_images")

    standard_scaler = StandardScaler()

    normalised_x_train_images = np.reshape(standard_scaler.fit_transform(np.reshape(x_train_images, (-1, 1))),
                                           x_train_images.shape)
    normalised_x_test_images = np.reshape(standard_scaler.transform(np.reshape(x_test_images, (-1, 1))),
                                          x_test_images.shape)

    return normalised_x_train_images, normalised_x_test_images, standard_scaler


def pad_images_array(images, padded_dimension_size):
    print("pad_images")

    padded_images = copy.deepcopy(images)

    while padded_images.shape[1] + 1 < padded_dimension_size:
        padded_images = np.pad(padded_images, ((0, 0), (1, 1), (0, 0), (0, 0)))

    if padded_images.shape[1] < padded_dimension_size:
        padded_images = np.pad(padded_images, ((0, 0), (0, 1), (0, 0), (0, 0)))

    while padded_images.shape[2] + 1 < padded_dimension_size:
        padded_images = np.pad(padded_images, ((0, 0,), (0, 0), (1, 1), (0, 0)))

    if padded_images.shape[2] < padded_dimension_size:
        padded_images = np.pad(padded_images, ((0, 0), (0, 0), (0, 1), (0, 0)))

    return padded_images


def preprocess_images_array(x_train_images, x_test_images):
    print("preprocess_images")

    x_train_images_preprocessed = np.array(x_train_images)
    x_test_images_preprocessed = np.array(x_test_images)

    x_train_images_preprocessed = rescale_images_array(x_train_images_preprocessed, output_dimension_size)
    x_test_images_preprocessed = rescale_images_array(x_test_images_preprocessed, output_dimension_size)

    x_train_images_preprocessed, x_test_images_preprocessed, standard_scaler = (
        normalise_images_array(x_train_images_preprocessed, x_test_images_preprocessed))

    x_train_original_shapes = []

    for i in range(x_train_images_preprocessed.shape[0]):
        x_train_original_shapes.append(x_train_images_preprocessed.shape[1:])

    x_test_original_shapes = []

    for i in range(x_test_images_preprocessed.shape[0]):
        x_test_original_shapes.append(x_test_images_preprocessed.shape[1:])

    x_train_padding_masks = np.ones(x_train_images_preprocessed.shape, dtype=np.float32)
    x_test_padding_masks = np.ones(x_test_images_preprocessed.shape, dtype=np.float32)

    x_train_images_preprocessed = pad_images_array(x_train_images_preprocessed, output_dimension_size)
    x_test_images_preprocessed = pad_images_array(x_test_images_preprocessed, output_dimension_size)

    x_train_padding_masks = pad_images_array(x_train_padding_masks, output_dimension_size)
    x_test_padding_masks = pad_images_array(x_test_padding_masks, output_dimension_size)

    x_train_images_preprocessed = tf.convert_to_tensor(x_train_images_preprocessed)
    x_test_images_preprocessed = tf.convert_to_tensor(x_test_images_preprocessed)

    x_train_padding_masks = x_train_padding_masks.astype(bool)
    x_test_padding_masks = x_test_padding_masks.astype(bool)

    x_train_padding_masks = tf.convert_to_tensor(x_train_padding_masks)
    x_test_padding_masks = tf.convert_to_tensor(x_test_padding_masks)

    return (x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks, x_test_images_preprocessed,
            x_test_original_shapes, x_test_padding_masks, standard_scaler)


def preprocess_labels(x_train_labels, x_test_labels):
    print("preprocess_labels")

    x_train_labels_preprocessed = copy.deepcopy(x_train_labels)
    x_test_labels_preprocessed = copy.deepcopy(x_test_labels)

    preprocessors = []

    if conditioning_input_embedding_bools[0]:
        distinct_clusters = len(np.unique(x_train_labels_preprocessed))

        current_number_of_label_bins = number_of_label_bins

        if current_number_of_label_bins > distinct_clusters:
            current_number_of_label_bins = distinct_clusters

        k_bins_discretiser = KBinsDiscretizer(n_bins=current_number_of_label_bins, encode="ordinal",
                                              strategy="kmeans", subsample=None)

        x_train_labels_preprocessed = np.reshape(k_bins_discretiser.fit_transform(
            np.reshape(x_train_labels_preprocessed, (-1, 1))), x_train_labels_preprocessed.shape)
        x_test_labels_preprocessed = np.reshape(k_bins_discretiser.transform(
            np.reshape(x_test_labels_preprocessed, (-1, 1))), x_test_labels_preprocessed.shape)

        preprocessors.append(k_bins_discretiser)
    else:
        standard_scaler = StandardScaler()

        x_train_labels_preprocessed = np.reshape(standard_scaler.fit_transform(
            np.reshape(x_train_labels_preprocessed, (-1, 1))), x_train_labels_preprocessed.shape)
        x_test_labels_preprocessed = np.reshape(standard_scaler.transform(
            np.reshape(x_test_labels_preprocessed, (-1, 1))), x_test_labels_preprocessed.shape)

        preprocessors.append(standard_scaler)

    x_train_labels_preprocessed = tf.convert_to_tensor(x_train_labels_preprocessed)
    x_test_labels_preprocessed = tf.convert_to_tensor(x_test_labels_preprocessed)

    return x_train_labels_preprocessed, x_test_labels_preprocessed, preprocessors


def preprocess_input(x_train_images, x_test_images, x_train_labels, x_test_labels):
    print("preprocess_input")

    x_train_images_preprocessed = copy.deepcopy(x_train_images)
    x_test_images_preprocessed = copy.deepcopy(x_test_images)

    if greyscale_bool:
        x_train_images_preprocessed = convert_rgb_to_greyscale(x_train_images_preprocessed)
        x_test_images_preprocessed = convert_rgb_to_greyscale(x_test_images_preprocessed)

    if read_data_from_storage_bool or preprocess_list_bool:
        (x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks, x_test_images_preprocessed,
         x_test_original_shapes, x_test_padding_masks, standard_scaler) = (
            preprocess_images_list(x_train_images_preprocessed, x_test_images_preprocessed))
    else:
        (x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks, x_test_images_preprocessed,
         x_test_original_shapes, x_test_padding_masks, standard_scaler) = (
            preprocess_images_array(x_train_images_preprocessed, x_test_images_preprocessed))

    x_train_labels_preprocessed, x_test_labels_preprocessed, preprocessors = preprocess_labels(x_train_labels,
                                                                                               x_test_labels)

    return (x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks, x_test_images_preprocessed,
            x_test_original_shapes, x_test_padding_masks, standard_scaler, x_train_labels_preprocessed,
            x_test_labels_preprocessed, preprocessors)


def get_previous_geometric_value(value, base):
    power = np.log2(value / base) + 1

    if not power.is_integer():
        previous_value = base * np.power(2.0, (np.floor(power) - 1.0))
    else:
        previous_value = copy.deepcopy(value)

    return previous_value


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

    x_latent = tf.keras.layers.Dense(units=units)(x)
    x = x_latent

    x = tf.keras.layers.Dense(units=x_flatten.shape[-1])(x)
    x = tf.keras.layers.Reshape(x_input.shape[1:])(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x, x_latent])

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

    units = get_previous_geometric_value(x.shape[-1] - 1, 2.0)

    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x_latent = tf.keras.layers.Dense(units=units)(x)
    x = x_latent

    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(dense_layers - 2, -1, -1):
        x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1] + 1, 2.0))(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=x_flatten.shape[-1])(x)
    x = tf.keras.layers.Reshape(x_input.shape[1:])(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x, x_latent])

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

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x_latent = tf.keras.layers.Conv2D(filters=latent_filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="same")(x)
    x = x_latent

    for i in range(conv_layers[-1]):
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
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x, x_latent])

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

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    if conditioning_input_embedding_bools[0]:
        x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                             output_dim=conditioning_inputs_size[0],
                                             embeddings_initializer=tf.keras.initializers.orthogonal)
                   (x_label_input))
        x_label = x_label[:, 0]
    else:
        x_label = tf.keras.layers.Dense(units=conditioning_inputs_size[0],
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label_input)

    x_conditioning = tf.keras.layers.Concatenate()([x_label])

    if np.any(conditioning_input_embedding_bools is False):
        for i in range(conditional_dense_layers):
            x_conditioning = tf.keras.layers.Dense(units=conditioning_inputs_size[1],
                                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_conditioning)
            x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

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
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_latent = tf.keras.layers.Conv2D(filters=latent_filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x)
    x = x_latent

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
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

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
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
                           outputs=[x, x_latent])

    return model


def get_model_conv_alex_gaussian_latent(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    latent_hw_size = int(np.round(image_input_shape[0] / np.power(2.0, len(filters) - 1.0)))
    latent_shape = (latent_hw_size, latent_hw_size, latent_filters)

    x_latent_gaussian_input = tf.keras.Input(shape=latent_shape)

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    if conditioning_input_embedding_bools[0]:
        x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                             output_dim=conditioning_inputs_size[0],
                                             embeddings_initializer=tf.keras.initializers.orthogonal)
                   (x_label_input))
        x_label = x_label[:, 0]
    else:
        x_label = tf.keras.layers.Dense(units=conditioning_inputs_size[0],
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label_input)

    x_conditioning = tf.keras.layers.Concatenate()([x_label])

    if conditioning_input_embedding_bools[0]:
        for i in range(conditional_dense_layers):
            x_conditioning = tf.keras.layers.Dense(units=conditioning_inputs_size[1],
                                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_conditioning)
            x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

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
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape[-1] != x.shape[-1]:
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
        x_latent_mean = tf.keras.layers.GroupNormalization(groups=8,
                                                           center=False,
                                                           scale=False)(x_latent_mean)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_latent_mean = tf.keras.layers.Multiply()([x_latent_mean, x_conditioning_gamma])
        x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_conditioning_beta])

        x_latent_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_mean)

    if x_res.shape[-1] != x_latent_mean.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_latent_mean.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_res])

    x_latent_mean = tf.keras.layers.Conv2D(filters=latent_filters,
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
        x_latent_stddev = tf.keras.layers.GroupNormalization(groups=8,
                                                             center=False,
                                                             scale=False)(x_latent_stddev)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_latent_stddev = tf.keras.layers.Multiply()([x_latent_stddev, x_conditioning_gamma])
        x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_conditioning_beta])

        x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_stddev)

    if x_res.shape[-1] != x_latent_stddev.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_latent_stddev.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_res])

    x_latent_stddev = tf.keras.layers.Conv2D(filters=latent_filters,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_stddev)

    x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_latent_stddev)
    x_latent_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_latent_stddev)
    x_latent_stddev_half = tf.keras.layers.Lambda(lambda x_current: x_current * 0.5)(x_latent_stddev)

    x = tf.keras.layers.Multiply()([x_latent_gaussian_input, x_latent_stddev_half])
    x = tf.keras.layers.Add()([x, x_latent_mean])

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
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

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
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

    if conditioning_input_embedding_bools[0]:
        x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                             output_dim=conditioning_inputs_size[0],
                                             embeddings_initializer=tf.keras.initializers.orthogonal)
                   (x_label_input))
        x_label = x_label[:, 0]
    else:
        x_label = tf.keras.layers.Dense(units=conditioning_inputs_size[0],
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label_input)

    x_conditioning = tf.keras.layers.Concatenate()([x_label])

    if np.any(conditioning_input_embedding_bools is False):
        for i in range(conditional_dense_layers):
            x_conditioning = tf.keras.layers.Dense(units=conditioning_inputs_size[1],
                                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_conditioning)
            x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

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
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_latent = tf.keras.layers.Conv2D(filters=latent_filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x)
    x_latent_shape_prod = tf.math.reduce_prod(x_latent.shape[1:-1]).numpy()
    x_latent_quantised, x_latent_discretised = VectorQuantiser(embedding_dim=x_latent_shape_prod,
                                                               num_embeddings=num_embeddings)(x_latent)

    x = x_latent_quantised

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
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

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
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

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    if conditioning_input_embedding_bools[0]:
        x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                             output_dim=conditioning_inputs_size[0],
                                             embeddings_initializer=tf.keras.initializers.orthogonal)
                   (x_label_input))
        x_label = x_label[:, 0]
    else:
        x_label = tf.keras.layers.Dense(units=conditioning_inputs_size[0],
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label_input)

    x_conditioning = tf.keras.layers.Concatenate()([x_label])

    if np.any(conditioning_input_embedding_bools is False):
        for i in range(conditional_dense_layers):
            x_conditioning = tf.keras.layers.Dense(units=conditioning_inputs_size[1],
                                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_conditioning)
            x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

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
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_latent = tf.keras.layers.Conv2D(filters=latent_filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x)
    x = x_latent

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
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

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_mean = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_mean = tf.keras.layers.Conv2D(filters=filters[0],
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)
        x_mean = tf.keras.layers.GroupNormalization(groups=8,
                                                    center=False,
                                                    scale=False)(x_mean)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_mean = tf.keras.layers.Multiply()([x_mean, x_conditioning_gamma])
        x_mean = tf.keras.layers.Add()([x_mean, x_conditioning_beta])

        x_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_mean)

    if x_res.shape[-1] != x_mean.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_mean.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_mean = tf.keras.layers.Add()([x_mean, x_res])

    x_mean = tf.keras.layers.Conv2D(filters=latent_filters,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)

    x_stddev = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_stddev = tf.keras.layers.Conv2D(filters=filters[0],
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)
        x_stddev = tf.keras.layers.GroupNormalization(groups=8,
                                                      center=False,
                                                      scale=False)(x_stddev)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_stddev = tf.keras.layers.Multiply()([x_stddev, x_conditioning_gamma])
        x_stddev = tf.keras.layers.Add()([x_stddev, x_conditioning_beta])

        x_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_stddev)

    if x_res.shape[-1] != x_stddev.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_stddev.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_stddev = tf.keras.layers.Add()([x_stddev, x_res])

    x_stddev = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)
    x_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_stddev)
    x_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_stddev)

    model = tf.keras.Model(inputs=[x_image_input, x_label_input],
                           outputs=[x_mean, x_stddev, x_latent])

    return model


def get_model_conv_alex_gaussian_latent_gaussian_negative_log_likelihood(x_train_images, x_train_labels):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train_images[0])

        image_input_shape = list(current_x_train_images.shape)
    else:
        image_input_shape = list(x_train_images.shape[1:])

    x_image_input = tf.keras.Input(shape=image_input_shape)
    x_label_input = tf.keras.Input(shape=x_train_labels.shape[1:])

    latent_hw_size = int(np.round(image_input_shape[0] / np.power(2.0, len(filters) - 1.0)))
    latent_shape = (latent_hw_size, latent_hw_size, latent_filters)

    x_latent_gaussian_input = tf.keras.Input(shape=latent_shape)

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_image_input)

    if conditioning_input_embedding_bools[0]:
        x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                             output_dim=conditioning_inputs_size[0],
                                             embeddings_initializer=tf.keras.initializers.orthogonal)
                   (x_label_input))
        x_label = x_label[:, 0]
    else:
        x_label = tf.keras.layers.Dense(units=conditioning_inputs_size[0],
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label_input)

    x_conditioning = tf.keras.layers.Concatenate()([x_label])

    if np.any(conditioning_input_embedding_bools is False):
        for i in range(conditional_dense_layers):
            x_conditioning = tf.keras.layers.Dense(units=conditioning_inputs_size[1],
                                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_conditioning)
            x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

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
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape[-1] != x.shape[-1]:
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
        x_latent_mean = tf.keras.layers.GroupNormalization(groups=8,
                                                           center=False,
                                                           scale=False)(x_latent_mean)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_latent_mean = tf.keras.layers.Multiply()([x_latent_mean, x_conditioning_gamma])
        x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_conditioning_beta])

        x_latent_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_mean)

    if x_res.shape[-1] != x_latent_mean.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_latent_mean.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_latent_mean = tf.keras.layers.Add()([x_latent_mean, x_res])

    x_latent_mean = tf.keras.layers.Conv2D(filters=latent_filters,
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
        x_latent_stddev = tf.keras.layers.GroupNormalization(groups=8,
                                                             center=False,
                                                             scale=False)(x_latent_stddev)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_latent_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_latent_stddev = tf.keras.layers.Multiply()([x_latent_stddev, x_conditioning_gamma])
        x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_conditioning_beta])

        x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_latent_stddev)

    if x_res.shape[-1] != x_latent_stddev.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_latent_stddev.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_latent_stddev = tf.keras.layers.Add()([x_latent_stddev, x_res])

    x_latent_stddev = tf.keras.layers.Conv2D(filters=latent_filters,
                                             kernel_size=(3, 3),
                                             strides=(1, 1),
                                             padding="same",
                                             kernel_initializer=tf.keras.initializers.orthogonal)(x_latent_stddev)

    x_latent_stddev = tf.keras.layers.Lambda(tf.keras.activations.softplus)(x_latent_stddev)
    x_latent_stddev = tf.keras.layers.Lambda(lambda x_current: x_current + tf.keras.backend.epsilon())(x_latent_stddev)
    x_latent_stddev_half = tf.keras.layers.Lambda(lambda x_current: x_current * 0.5)(x_latent_stddev)

    x = tf.keras.layers.Multiply()([x_latent_gaussian_input, x_latent_stddev_half])
    x = tf.keras.layers.Add()([x, x_latent_mean])

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
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

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_mean = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_mean = tf.keras.layers.Conv2D(filters=filters[0],
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)
        x_mean = tf.keras.layers.GroupNormalization(groups=8,
                                                    center=False,
                                                    scale=False)(x_mean)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_mean = tf.keras.layers.Multiply()([x_mean, x_conditioning_gamma])
        x_mean = tf.keras.layers.Add()([x_mean, x_conditioning_beta])

        x_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_mean)

    if x_res.shape[-1] != x_mean.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_mean.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_mean = tf.keras.layers.Add()([x_mean, x_res])

    x_mean = tf.keras.layers.Conv2D(filters=latent_filters,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)

    x_stddev = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_stddev = tf.keras.layers.Conv2D(filters=filters[0],
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)
        x_stddev = tf.keras.layers.GroupNormalization(groups=8,
                                                      center=False,
                                                      scale=False)(x_stddev)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_stddev = tf.keras.layers.Multiply()([x_stddev, x_conditioning_gamma])
        x_stddev = tf.keras.layers.Add()([x_stddev, x_conditioning_beta])

        x_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_stddev)

    if x_res.shape[-1] != x_stddev.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_stddev.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_stddev = tf.keras.layers.Add()([x_stddev, x_res])

    x_stddev = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                      kernel_size=(1, 1),
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

    if conditioning_input_embedding_bools[0]:
        x_label = (tf.keras.layers.Embedding(input_dim=int(tf.math.round(tf.reduce_max(x_train_labels) + 1.0)),
                                             output_dim=conditioning_inputs_size[0],
                                             embeddings_initializer=tf.keras.initializers.orthogonal)
                   (x_label_input))
        x_label = x_label[:, 0]
    else:
        x_label = tf.keras.layers.Dense(units=conditioning_inputs_size[0],
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_label_input)

    x_conditioning = tf.keras.layers.Concatenate()([x_label])

    if np.any(conditioning_input_embedding_bools is False):
        for i in range(conditional_dense_layers):
            x_conditioning = tf.keras.layers.Dense(units=conditioning_inputs_size[1],
                                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_conditioning)
            x_conditioning = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

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
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x_res = x

        x = tf.keras.layers.GroupNormalization(groups=1)(x)

        x_shape = x.shape
        x_shape_prod = tf.math.reduce_prod(x_shape[1:-1]).numpy()
        x = tf.keras.layers.Reshape(target_shape=(x_shape_prod, x_shape[-1]))(x)
        x = (tfm.nlp.layers.KernelAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            kernel_initializer=tf.keras.initializers.orthogonal,
                                            feature_transform="elu")(x, x))
        x = tf.keras.layers.Reshape(target_shape=x_shape[1:])(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_latent = tf.keras.layers.Conv2D(filters=latent_filters,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding="same",
                                      kernel_initializer=tf.keras.initializers.orthogonal)(x)
    x_latent_shape_prod = tf.math.reduce_prod(x_latent.shape[1:-1]).numpy()
    x_latent_quantised, x_latent_discretised = VectorQuantiser(embedding_dim=x_latent_shape_prod,
                                                               num_embeddings=num_embeddings)(x_latent)

    x = x_latent_quantised

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.GroupNormalization(groups=8,
                                               center=False,
                                               scale=False)(x)

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
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
        x = tf.keras.layers.Add()([x, x_conditioning_beta])

        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    if x_res.shape[-1] != x.shape[-1]:
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

        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8,
                                                   center=False,
                                                   scale=False)(x)

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
            x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

            x = tf.keras.layers.Multiply()([x, x_conditioning_gamma])
            x = tf.keras.layers.Add()([x, x_conditioning_beta])

            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        if x_res.shape[-1] != x.shape[-1]:
            x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding="same",
                                           kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

        x = tf.keras.layers.Add()([x, x_res])

    x_mean = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_mean = tf.keras.layers.Conv2D(filters=filters[0],
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        padding="same",
                                        kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)
        x_mean = tf.keras.layers.GroupNormalization(groups=8,
                                                    center=False,
                                                    scale=False)(x_mean)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_mean.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_mean = tf.keras.layers.Multiply()([x_mean, x_conditioning_gamma])
        x_mean = tf.keras.layers.Add()([x_mean, x_conditioning_beta])

        x_mean = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_mean)

    if x_res.shape[-1] != x_mean.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_mean.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_mean = tf.keras.layers.Add()([x_mean, x_res])

    x_mean = tf.keras.layers.Conv2D(filters=latent_filters,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=tf.keras.initializers.orthogonal)(x_mean)

    x_stddev = x
    x_res = x

    for i in range(conv_layers[-1]):
        x_stddev = tf.keras.layers.Conv2D(filters=filters[0],
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same",
                                          kernel_initializer=tf.keras.initializers.orthogonal)(x_stddev)
        x_stddev = tf.keras.layers.GroupNormalization(groups=8,
                                                      center=False,
                                                      scale=False)(x_stddev)

        x_conditioning_beta = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning))
        x_conditioning_beta = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_beta)
        x_conditioning_beta = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                     kernel_initializer=tf.keras.initializers.orthogonal)
                               (x_conditioning_beta))

        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning))
        x_conditioning_gamma = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_conditioning_gamma)
        x_conditioning_gamma = (tf.keras.layers.Dense(units=x_stddev.shape[-1],
                                                      kernel_initializer=tf.keras.initializers.orthogonal)
                                (x_conditioning_gamma))
        x_conditioning_gamma = tf.keras.layers.Lambda(lambda x_current: x_current + 1.0)(x_conditioning_gamma)

        x_stddev = tf.keras.layers.Multiply()([x_stddev, x_conditioning_gamma])
        x_stddev = tf.keras.layers.Add()([x_stddev, x_conditioning_beta])

        x_stddev = tf.keras.layers.Lambda(tf.keras.activations.swish)(x_stddev)

    if x_res.shape[-1] != x_stddev.shape[-1]:
        x_res = tf.keras.layers.Conv2D(filters=x_stddev.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)

    x_stddev = tf.keras.layers.Add()([x_stddev, x_res])

    x_stddev = tf.keras.layers.Conv2D(filters=image_input_shape[-1],
                                      kernel_size=(1, 1),
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
    flipped_image = copy.deepcopy(image)

    if axis_zero_flip_bool:
        if random.choice([True, False]):
            flipped_image = np.flip(flipped_image, axis=0)

    if axis_one_flip_bool:
        if random.choice([True, False]):
            flipped_image = np.flip(flipped_image, axis=1)

    return flipped_image


def translate_image(image):
    max_translation = int(np.round(np.min(image.shape) * translate_proportion))

    translation = random.randint(0, max_translation)

    if random.choice([True, False]):
        translated_image = np.pad(image, ((0, translation), (0, 0), (0, 0)))
    else:
        translated_image = np.pad(image, ((translation, 0), (0, 0), (0, 0)))

    translation = random.randint(0, max_translation)

    if random.choice([True, False]):
        translated_image = np.pad(translated_image, ((0, 0), (0, translation), (0, 0)))
    else:
        translated_image = np.pad(translated_image, ((0, 0), (translation, 0), (0, 0)))

    return translated_image


def crop_image(image, cropped_dimension_size, output_shape=None):
    cropped_image = copy.deepcopy(image)

    if output_shape is not None:
        cropped_dimension_size = output_shape[0]

    while cropped_image.shape[0] - 1 > cropped_dimension_size:
        cropped_image = cropped_image[1:-1]

    if cropped_image.shape[0] > cropped_dimension_size:
        cropped_image = cropped_image[1:]

    if output_shape is not None:
        cropped_dimension_size = output_shape[1]

    while cropped_image.shape[1] - 1 > cropped_dimension_size:
        cropped_image = cropped_image[:, 1:-1]

    if cropped_image.shape[1] > cropped_dimension_size:
        cropped_image = cropped_image[:, 1:]

    return cropped_image


def augmentation(image, original_shape, padding_mask):
    augmented_image = image.numpy()

    if augmentation_bool:
        input_shape = augmented_image.shape

        augmented_image = augmented_image[padding_mask]
        augmented_image = np.reshape(augmented_image, original_shape)

        augmented_image = flip_image(augmented_image)

        if gaussian_bool:
            augmented_image = gaussian(augmented_image, sigma=random.uniform(0.0, max_sigma), mode="constant",
                                       preserve_range=True, channel_axis=-1)

        if sharpen_bool:
            augmented_image = unsharp_mask(augmented_image, radius=random.uniform(0.0, max_radius),
                                           amount=random.uniform(0.0, max_amount), preserve_range=True,
                                           channel_axis=-1)

        if scale_bool:
            augmented_image = rescale(augmented_image, random.uniform(min_scale, max_scale), order=3,
                                      preserve_range=True, channel_axis=-1)

        if rotate_bool:
            augmented_image = scipy.ndimage.rotate(augmented_image, angle=random.uniform(min_angle, max_angle),
                                                   axes=(0, 1), order=1)

        if translate_bool:
            augmented_image = translate_image(augmented_image)

        augmented_image = pad_image(augmented_image, None, original_shape)
        augmented_image = crop_image(augmented_image, None, original_shape)

        augmented_image = pad_image(augmented_image, None, input_shape)

    augmented_image = tf.convert_to_tensor(augmented_image)

    return augmented_image


def mean_squared_error(y_true, y_pred, mask):
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2.0))


def root_mean_squared_error(y_true, y_pred, mask):
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.math.sqrt(tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2.0)) + tf.keras.backend.epsilon())


def gaussian_negative_log_likelihood(y_true, y_pred_mean, y_pred_std, mask):
    y_true = tf.boolean_mask(y_true, mask)
    y_pred_mean = tf.boolean_mask(y_pred_mean, mask)
    y_pred_std = tf.boolean_mask(y_pred_std, mask)

    y_true = tf.reshape(y_true, (y_true.shape[0], -1))
    y_pred_mean = tf.reshape(y_pred_mean, (y_pred_mean.shape[0], -1))
    y_pred_std = tf.reshape(y_pred_std, (y_pred_std.shape[0], -1))

    return -tf.math.reduce_mean((-(tf.math.pow(
        (tf.math.divide_no_nan((y_true - y_pred_mean), y_pred_std)), 2.0) / 2.0) -
                                 (tf.math.log(y_pred_std + tf.keras.backend.epsilon())) -
                                 (tf.math.log(2.0 * np.pi) / 2.0)))


def forward_gaussian_kullback_leibler_divergence(y_pred_mean, y_pred_std):
    y_pred_mean = tf.reshape(y_pred_mean, (y_pred_mean.shape[0], -1))
    y_pred_std = tf.reshape(y_pred_std, (y_pred_std.shape[0], -1))

    return tf.math.reduce_mean(0.5 * ((tf.math.pow(
        tf.math.log(y_pred_std + tf.keras.backend.epsilon()),
        2.0) + tf.math.divide_no_nan((1.0 + tf.math.pow(-y_pred_mean, 2.0)),
                                     tf.math.pow(y_pred_std, 2.0))) - 1.0))


def reversed_gaussian_kullback_leibler_divergence(y_pred_mean, y_pred_std):
    y_pred_mean = tf.reshape(y_pred_mean, (y_pred_mean.shape[0], -1))
    y_pred_std = tf.reshape(y_pred_std, (y_pred_std.shape[0], -1))

    return tf.math.reduce_mean(0.5 * ((tf.math.pow(
        tf.math.log(tf.math.divide_no_nan(1.0, y_pred_std) + tf.keras.backend.epsilon()),
        2.0) + (tf.math.pow(y_pred_std, 2.0) + tf.math.pow(y_pred_mean, 2.0))) - 1.0))


def get_error(y_true, y_pred, mask):
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.math.reduce_mean(tf.math.abs(tf.math.log(tf.math.abs(tf.math.divide_no_nan(y_pred, y_true)) +
                                                       tf.keras.backend.epsilon()))) * 100.0


def output_image(image, original_shape, padding_mask, standard_scaler, current_output_path):
    outputted_image = image.numpy()

    if original_shape is not None and padding_mask is not None:
        outputted_image = outputted_image[padding_mask.numpy()]
        outputted_image = np.reshape(outputted_image, original_shape)

    outputted_image = np.reshape(standard_scaler.inverse_transform(np.reshape(outputted_image, (-1, 1))),
                                 outputted_image.shape)
    outputted_image = np.clip(np.round(outputted_image), 0.0, 255.0).astype(np.uint8)

    if greyscale_bool:
        outputted_image = outputted_image[:, :, 0]

    outputted_image = Image.fromarray(outputted_image)
    outputted_image.save(current_output_path)

    return True


def validate(model, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler, x_test_labels, i):
    validate_output_path = "{0}/validate/".format(output_path)

    output_input_bool = False

    if not os.path.exists(validate_output_path):
        mkdir_p(validate_output_path)

        output_input_bool = True

    current_x_test_image = get_data_from_storage(x_test_images[0])
    current_x_test_original_shape = x_test_original_shapes[0]
    current_x_test_padding_mask = get_data_from_storage(x_test_padding_masks[0])
    current_x_test_label = x_test_labels[0]

    current_x_test_image = tf.expand_dims(current_x_test_image, axis=0)
    current_x_test_padding_mask = tf.expand_dims(current_x_test_padding_mask, axis=0)
    current_x_test_label = tf.expand_dims(current_x_test_label, axis=0)

    latent_hw_size = int(np.round(current_x_test_image.shape[1] / np.power(2.0, len(filters) - 1.0)))
    latent_shape = (latent_hw_size, latent_hw_size, latent_filters)

    if conv_bool and alex_bool:
        if gaussian_negative_log_likelihood_bool:
            if gaussian_latent_bool:
                y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                    model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                          training=False))

                y_pred_means = [y_pred_mean]
                y_pred_stddevs = [y_pred_stddev]

                for j in range(int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0),
                                                       32.0])) - 1):
                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                              training=False))

                    y_pred_means.append(y_pred_mean)
                    y_pred_stddevs.append(y_pred_stddev)

                y_pred_mean = tf.math.reduce_mean(y_pred_means, axis=0)
                y_pred_stddev = tf.math.reduce_mean(y_pred_stddevs, axis=0)

                output_image(y_pred_mean[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                             standard_scaler, "{0}/{1}_mean.png".format(validate_output_path, str(i)))

                y_pred = []

                for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                    y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                y_pred = tf.math.reduce_mean(y_pred, axis=0)
            else:
                if discrete_bool:
                    y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                        model([current_x_test_image, current_x_test_label], training=False))

                    output_image(y_pred_mean[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                                 standard_scaler, "{0}/{1}_mean.png".format(validate_output_path, str(i)))

                    y_pred = []

                    for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                        y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                    y_pred = tf.math.reduce_mean(y_pred, axis=0)
                else:
                    y_pred_mean, y_pred_stddev, y_latent = model([current_x_test_image, current_x_test_label],
                                                                 training=False)

                    output_image(y_pred_mean[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                                 standard_scaler, "{0}/{1}_mean.png".format(validate_output_path, str(i)))

                    y_pred = []

                    for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                        y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                    y_pred = tf.math.reduce_mean(y_pred, axis=0)
        else:
            if gaussian_latent_bool:
                y_pred, y_latent_mean, y_latent_stddev = (
                    model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                          training=False))

                y_preds = [y_pred]

                for j in range(int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0),
                                                       32.0])) - 1):
                    y_pred, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                              training=False))

                    y_preds.append(y_pred)

                y_pred = tf.math.reduce_mean(y_preds, axis=0)
            else:
                if discrete_bool:
                    y_pred, x_latent_quantised, x_latent_discretised = (
                        model([current_x_test_image, current_x_test_label], training=False))
                else:
                    y_pred, y_latent = (model([current_x_test_image, current_x_test_label], training=False))
    else:
        y_pred, x_latent = model([current_x_test_image], training=False)

    output_image(y_pred[0], current_x_test_original_shape, current_x_test_padding_mask[0], standard_scaler,
                 "{0}/{1}.png".format(validate_output_path, str(i)))

    if output_input_bool:
        output_image(current_x_test_image[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                     standard_scaler, "{0}/input.png".format(validate_output_path))

    return True


def train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images,
                                x_train_output_shapes, x_train_padding_masks, x_test_images, x_test_original_shapes,
                                x_test_padding_masks, standard_scaler, x_train_labels, x_test_labels):
    print("train")

    validate(model, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler, x_test_labels, 0)

    x_train_images_len = len(x_train_images)

    current_batch_size = None
    indices = list(range(x_train_images_len))

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    if conv_bool and alex_bool and gaussian_latent_bool and not discrete_bool:
        latent_size_x_train_image = get_data_from_storage(x_train_images[0])

        latent_hw_size = int(np.round(latent_size_x_train_image.shape[1] / np.power(2.0, len(filters) - 1.0)))
        latent_shape = (latent_hw_size, latent_hw_size, latent_filters)
    else:
        latent_shape = None

    current_gaussian_latent_loss_weight = 1.0

    for i in range(epochs):
        if i + 1 > kullback_leibler_divergence_epochs:
            current_gaussian_latent_loss_weight = gaussian_latent_loss_weight

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
            current_gradient_accumulation_batch_size = (
                int(np.floor(current_batch_size / gradient_accumulation_batch_size)))

            losses = []
            errors = []
            errors_range = []

            for k in range(current_gradient_accumulation_batch_size):
                current_x_train_images = []
                current_x_train_padding_masks = []
                current_x_train_labels = []

                for n in range(gradient_accumulation_batch_size):
                    current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])
                    current_x_train_output_shape = x_train_output_shapes[indices[current_index]]
                    current_x_train_padding_mask = get_data_from_storage(x_train_padding_masks[indices[current_index]])

                    current_x_train_label = x_train_labels[indices[current_index]]

                    current_x_train_image = augmentation(current_x_train_image, current_x_train_output_shape,
                                                         current_x_train_padding_mask)

                    current_x_train_images.append(current_x_train_image)
                    current_x_train_padding_masks.append(current_x_train_padding_mask)

                    current_x_train_labels.append(current_x_train_label)

                    current_index = current_index + 1

                current_x_train_images = tf.convert_to_tensor(current_x_train_images)
                current_x_train_padding_masks = tf.convert_to_tensor(current_x_train_padding_masks)
                current_x_train_labels = tf.convert_to_tensor(current_x_train_labels)

                if conv_bool and alex_bool:
                    if gaussian_negative_log_likelihood_bool:
                        if gaussian_latent_bool:
                            if i + 1 > mean_squared_error_epochs:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_images, current_x_train_labels,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    loss = tf.math.reduce_sum([
                                        gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                         y_pred_stddev, current_x_train_padding_masks),
                                        current_gaussian_latent_loss_weight *
                                        reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_images, current_x_train_labels,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                    loss = tf.math.reduce_sum([
                                        mean_squared_error(current_x_train_images, y_pred,
                                                           current_x_train_padding_masks),
                                        current_gaussian_latent_loss_weight *
                                        reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])
                        else:
                            if discrete_bool:
                                if i + 1 > discrete_mean_squared_error_epochs:
                                    if i + 1 > mean_squared_error_epochs:
                                        with tf.GradientTape() as tape:
                                            y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                                model([current_x_train_images, current_x_train_labels], training=True))

                                            loss = tf.math.reduce_sum([
                                                gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                                 y_pred_stddev,
                                                                                 current_x_train_padding_masks),
                                                tf.math.reduce_sum(model.losses)])
                                    else:
                                        with tf.GradientTape() as tape:
                                            y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                                model([current_x_train_images, current_x_train_labels], training=True))

                                            y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                            loss = tf.math.reduce_sum([
                                                mean_squared_error(current_x_train_images, y_pred,
                                                                   current_x_train_padding_masks),
                                                tf.math.reduce_sum(model.losses)])
                                else:
                                    with tf.GradientTape() as tape:
                                        y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                        loss = tf.math.reduce_sum([mean_squared_error(y_pred, y_pred,
                                                                                      current_x_train_padding_masks),
                                                                   tf.math.reduce_sum(model.losses)])
                            else:
                                if i + 1 > mean_squared_error_epochs:
                                    with tf.GradientTape() as tape:
                                        y_pred_mean, y_pred_stddev, y_latent = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([
                                            gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                             y_pred_stddev,
                                                                             current_x_train_padding_masks),
                                            tf.math.reduce_sum(model.losses)])
                                else:
                                    with tf.GradientTape() as tape:
                                        y_pred_mean, y_pred_stddev, y_latent = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                        loss = tf.math.reduce_sum([
                                            mean_squared_error(current_x_train_images, y_pred,
                                                               current_x_train_padding_masks),
                                            tf.math.reduce_sum(model.losses)])

                        y_pred_stddev_full_coverage = y_pred_stddev * 3.0

                        error_bounds = [get_error(current_x_train_images, y_pred_mean + y_pred_stddev_full_coverage,
                                                  current_x_train_padding_masks),
                                        get_error(current_x_train_images, y_pred_mean - y_pred_stddev_full_coverage,
                                                  current_x_train_padding_masks)]
                        error_upper_bound = tf.reduce_max(error_bounds)
                        error_lower_bound = tf.reduce_min(error_bounds)

                        errors.append(tf.math.reduce_mean([error_upper_bound, error_lower_bound]))
                        errors_range.append(error_upper_bound - error_lower_bound)
                    else:
                        if gaussian_latent_bool:
                            if i + 1 > mean_squared_error_epochs:
                                with tf.GradientTape() as tape:
                                    y_pred, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_images, current_x_train_labels,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    loss = tf.math.reduce_sum([
                                        root_mean_squared_error(current_x_train_images, y_pred,
                                                                current_x_train_padding_masks),
                                        current_gaussian_latent_loss_weight *
                                        reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred, y_latent_mean, y_latent_stddev = (
                                        model([current_x_train_images, current_x_train_labels,
                                               tf.random.normal((1,) + latent_shape)], training=True))

                                    loss = tf.math.reduce_sum([
                                        mean_squared_error(current_x_train_images, y_pred,
                                                           current_x_train_padding_masks),
                                        current_gaussian_latent_loss_weight *
                                        reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                        tf.math.reduce_sum(model.losses)])
                        else:
                            if discrete_bool:
                                if i + 1 > discrete_mean_squared_error_epochs:
                                    if i + 1 > mean_squared_error_epochs:
                                        with tf.GradientTape() as tape:
                                            y_pred, x_latent_quantised, x_latent_discretised = (
                                                model([current_x_train_images, current_x_train_labels], training=True))

                                            loss = tf.math.reduce_sum([
                                                root_mean_squared_error(current_x_train_images, y_pred,
                                                                        current_x_train_padding_masks),
                                                tf.math.reduce_sum(model.losses)])
                                    else:
                                        with tf.GradientTape() as tape:
                                            y_pred, x_latent_quantised, x_latent_discretised = (
                                                model([current_x_train_images, current_x_train_labels], training=True))

                                            loss = tf.math.reduce_sum([
                                                mean_squared_error(current_x_train_images, y_pred,
                                                                   current_x_train_padding_masks),
                                                tf.math.reduce_sum(model.losses)])
                                else:
                                    with tf.GradientTape() as tape:
                                        y_pred, x_latent_quantised, x_latent_discretised = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([mean_squared_error(y_pred, y_pred,
                                                                                      current_x_train_padding_masks),
                                                                   tf.math.reduce_sum(model.losses)])
                            else:
                                if i + 1 > mean_squared_error_epochs:
                                    with tf.GradientTape() as tape:
                                        y_pred, y_latent = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([
                                            root_mean_squared_error(current_x_train_images, y_pred,
                                                                    current_x_train_padding_masks),
                                            tf.math.reduce_sum(model.losses)])
                                else:
                                    with tf.GradientTape() as tape:
                                        y_pred, y_latent = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([
                                            mean_squared_error(current_x_train_images, y_pred,
                                                               current_x_train_padding_masks),
                                            tf.math.reduce_sum(model.losses)])

                        errors.append(get_error(current_x_train_images, y_pred, current_x_train_padding_masks))
                        errors_range.append(tf.constant(0.0))
                else:
                    with tf.GradientTape() as tape:
                        y_pred, x_latent = model([current_x_train_images], training=True)

                        loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred,
                                                                      current_x_train_padding_masks),
                                                   tf.math.reduce_sum(model.losses)])

                    errors.append(get_error(current_x_train_images, y_pred, current_x_train_padding_masks))
                    errors_range.append(tf.constant(0.0))

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)

            gradients = [gradient / current_batch_size for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            error = tf.math.reduce_mean(errors)
            error_range = tf.math.reduce_mean(errors_range)

            if alex_bool and i + 1 > mean_squared_error_epochs:
                if gaussian_negative_log_likelihood_bool:
                    loss_name = "Gaussian NLL"
                else:
                    loss_name = "RMSE"
            else:
                loss_name = "MSE"

            print(
                "Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss ({5}): {6:18} Error: {7:18} +/- {8:18}%".format(
                    str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), loss_name,
                    str(loss.numpy()), str(error.numpy()), str(error_range.numpy())))

        if use_ema and ema_overwrite_frequency is None:
            optimiser.finalize_variable_values(model.trainable_variables)

        validate(model, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler, x_test_labels,
                 i + 1)

    return model


def train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images, x_train_output_shapes,
          x_train_padding_masks, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler,
          x_train_labels, x_test_labels):
    print("train")

    validate(model, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler, x_test_labels, 0)

    x_train_images_len = len(x_train_images)

    current_batch_size = None
    indices = list(range(x_train_images_len))

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    if conv_bool and alex_bool and gaussian_latent_bool and not discrete_bool:
        latent_size_x_train_image = get_data_from_storage(x_train_images[0])

        latent_hw_size = int(np.round(latent_size_x_train_image.shape[1] / np.power(2.0, len(filters) - 1.0)))
        latent_shape = (latent_hw_size, latent_hw_size, latent_filters)
    else:
        latent_shape = None

    current_gaussian_latent_loss_weight = 0.0

    for i in range(epochs):
        if i + 1 > kullback_leibler_divergence_epochs:
            current_gaussian_latent_loss_weight = gaussian_latent_loss_weight

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
                current_x_train_output_shape = x_train_output_shapes[indices[current_index]]
                current_x_train_padding_mask = get_data_from_storage(x_train_padding_masks[indices[current_index]])

                current_x_train_label = x_train_labels[indices[current_index]]

                current_x_train_image = augmentation(current_x_train_image, current_x_train_output_shape,
                                                     current_x_train_padding_mask)

                current_x_train_images.append(current_x_train_image)
                current_x_train_padding_masks.append(current_x_train_padding_mask)

                current_x_train_labels.append(current_x_train_label)

                current_index = current_index + 1

            current_x_train_images = tf.convert_to_tensor(current_x_train_images)
            current_x_train_padding_masks = tf.convert_to_tensor(current_x_train_padding_masks)
            current_x_train_labels = tf.convert_to_tensor(current_x_train_labels)

            if conv_bool and alex_bool:
                if gaussian_negative_log_likelihood_bool:
                    if gaussian_latent_bool:
                        if i + 1 > mean_squared_error_epochs:
                            with tf.GradientTape() as tape:
                                y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)],
                                          training=True))

                                loss = tf.math.reduce_sum([
                                    gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                     y_pred_stddev, current_x_train_padding_masks),
                                    current_gaussian_latent_loss_weight *
                                    reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)], training=True))

                                y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                loss = tf.math.reduce_sum([
                                    mean_squared_error(current_x_train_images, y_pred, current_x_train_padding_masks),
                                    current_gaussian_latent_loss_weight *
                                    reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])
                    else:
                        if discrete_bool:
                            if i + 1 > discrete_mean_squared_error_epochs:
                                if i + 1 > mean_squared_error_epochs:
                                    with tf.GradientTape() as tape:
                                        y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([
                                            gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                             y_pred_stddev,
                                                                             current_x_train_padding_masks),
                                            tf.math.reduce_sum(model.losses)])
                                else:
                                    with tf.GradientTape() as tape:
                                        y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                        loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred,
                                                                                      current_x_train_padding_masks),
                                                                   tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, x_latent_quantised, x_latent_discretised = (
                                        model([current_x_train_images, current_x_train_labels], training=True))

                                    y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                    loss = tf.math.reduce_sum([mean_squared_error(y_pred, y_pred,
                                                                                  current_x_train_padding_masks),
                                                               tf.math.reduce_sum(model.losses)])
                        else:
                            if i + 1 > mean_squared_error_epochs:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, y_latent = (
                                        model([current_x_train_images, current_x_train_labels], training=True))

                                    loss = tf.math.reduce_sum([
                                        gaussian_negative_log_likelihood(current_x_train_images, y_pred_mean,
                                                                         y_pred_stddev, current_x_train_padding_masks),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred_mean, y_pred_stddev, y_latent = (
                                        model([current_x_train_images, current_x_train_labels], training=True))

                                    y_pred = tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev)

                                    loss = tf.math.reduce_sum([
                                        mean_squared_error(current_x_train_images, y_pred,
                                                           current_x_train_padding_masks),
                                        tf.math.reduce_sum(model.losses)])

                    y_pred_stddev_full_coverage = y_pred_stddev * 3.0

                    error_bounds = [get_error(current_x_train_images, y_pred_mean + y_pred_stddev_full_coverage,
                                              current_x_train_padding_masks),
                                    get_error(current_x_train_images, y_pred_mean - y_pred_stddev_full_coverage,
                                              current_x_train_padding_masks)]
                    error_upper_bound = tf.reduce_max(error_bounds)
                    error_lower_bound = tf.reduce_min(error_bounds)

                    error = get_error(current_x_train_images, y_pred_mean, current_x_train_padding_masks)
                    error_range = error_upper_bound - error_lower_bound
                else:
                    if gaussian_latent_bool:
                        if i + 1 > mean_squared_error_epochs:
                            with tf.GradientTape() as tape:
                                y_pred, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)],
                                          training=True))

                                loss = tf.math.reduce_sum([
                                    root_mean_squared_error(current_x_train_images, y_pred,
                                                            current_x_train_padding_masks),
                                    current_gaussian_latent_loss_weight *
                                    reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])
                        else:
                            with tf.GradientTape() as tape:
                                y_pred, y_latent_mean, y_latent_stddev = (
                                    model([current_x_train_images, current_x_train_labels,
                                           tf.random.normal((current_batch_size,) + latent_shape)], training=True))

                                loss = tf.math.reduce_sum([
                                    mean_squared_error(current_x_train_images, y_pred, current_x_train_padding_masks),
                                    current_gaussian_latent_loss_weight *
                                    reversed_gaussian_kullback_leibler_divergence(y_latent_mean, y_latent_stddev),
                                    tf.math.reduce_sum(model.losses)])
                    else:
                        if discrete_bool:
                            if i + 1 > discrete_mean_squared_error_epochs:
                                if i + 1 > mean_squared_error_epochs:
                                    with tf.GradientTape() as tape:
                                        y_pred, x_latent_quantised, x_latent_discretised = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([
                                            root_mean_squared_error(current_x_train_images, y_pred,
                                                                    current_x_train_padding_masks),
                                            tf.math.reduce_sum(model.losses)])
                                else:
                                    with tf.GradientTape() as tape:
                                        y_pred, x_latent_quantised, x_latent_discretised = (
                                            model([current_x_train_images, current_x_train_labels], training=True))

                                        loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred,
                                                                                      current_x_train_padding_masks),
                                                                   tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred, x_latent_quantised, x_latent_discretised = (
                                        model([current_x_train_images, current_x_train_labels], training=True))

                                    loss = tf.math.reduce_sum([mean_squared_error(y_pred, y_pred,
                                                                                  current_x_train_padding_masks),
                                                               tf.math.reduce_sum(model.losses)])
                        else:
                            if i + 1 > mean_squared_error_epochs:
                                with tf.GradientTape() as tape:
                                    y_pred, y_latent = model([current_x_train_images, current_x_train_labels],
                                                             training=True)

                                    loss = tf.math.reduce_sum([
                                        root_mean_squared_error(current_x_train_images, y_pred,
                                                                current_x_train_padding_masks),
                                        tf.math.reduce_sum(model.losses)])
                            else:
                                with tf.GradientTape() as tape:
                                    y_pred, y_latent = model([current_x_train_images, current_x_train_labels],
                                                             training=True)

                                    loss = tf.math.reduce_sum([
                                        mean_squared_error(current_x_train_images, y_pred,
                                                           current_x_train_padding_masks),
                                        tf.math.reduce_sum(model.losses)])

                    error = get_error(current_x_train_images, y_pred, current_x_train_padding_masks)
                    error_range = tf.constant(0.0)
            else:
                with tf.GradientTape() as tape:
                    y_pred, x_latent = model([current_x_train_images], training=True)

                    loss = tf.math.reduce_sum([mean_squared_error(current_x_train_images, y_pred,
                                                                  current_x_train_padding_masks),
                                               tf.math.reduce_sum(model.losses)])

                error = get_error(current_x_train_images, y_pred, current_x_train_padding_masks)
                error_range = tf.constant(0.0)

            gradients = tape.gradient(loss, model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            if alex_bool and i + 1 > mean_squared_error_epochs:
                if gaussian_negative_log_likelihood_bool:
                    loss_name = "Gaussian NLL"
                else:
                    loss_name = "RMSE"
            else:
                loss_name = "MSE"

            print(
                "Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss ({5}): {6:18} Error: {7:18} +/- {8:18}%".format(
                    str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), loss_name,
                    str(loss.numpy()), str(error.numpy()), str(error_range.numpy())))

        if use_ema and ema_overwrite_frequency is None:
            optimiser.finalize_variable_values(model.trainable_variables)

        validate(model, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler, x_test_labels,
                 i + 1)

    return model


def test(model, x_images, x_original_shapes, x_padding_masks, standard_scaler, x_labels, output_prefix):
    print("test")

    test_output_path = "{0}/test/{1}/".format(output_path, output_prefix)
    mkdir_p(test_output_path)

    for i in range(len(x_images)):
        current_test_output_path = "{0}/{1}/".format(test_output_path, str(i))
        mkdir_p(current_test_output_path)

        current_x_test_image = get_data_from_storage(x_images[i])
        current_x_test_original_shape = x_original_shapes[i]
        current_x_test_padding_mask = get_data_from_storage(x_padding_masks[i])
        current_x_test_label = x_labels[i]

        current_x_test_image = tf.convert_to_tensor(current_x_test_image)
        current_x_test_padding_mask = tf.convert_to_tensor(current_x_test_padding_mask)

        current_x_test_image = tf.expand_dims(current_x_test_image, axis=0)
        current_x_test_padding_mask = tf.expand_dims(current_x_test_padding_mask, axis=0)
        current_x_test_label = tf.expand_dims(current_x_test_label, axis=0)

        latent_hw_size = int(np.round(current_x_test_image.shape[1] / np.power(2.0, len(filters) - 1.0)))
        latent_shape = (latent_hw_size, latent_hw_size, latent_filters)

        if alex_bool:
            if gaussian_negative_log_likelihood_bool:
                if gaussian_latent_bool:
                    y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                              training=False))

                    output_image(y_latent_mean[0], None, None, standard_scaler,
                                 "{0}/latent_mean.png".format(current_test_output_path))

                    output_image(y_latent_stddev[0], None, None, standard_scaler,
                                 "{0}/latent_stddev.png".format(current_test_output_path))

                    y_pred_means = [y_pred_mean]
                    y_pred_stddevs = [y_pred_stddev]

                    for k in range(int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0),
                                                           32.0])) - 1):
                        y_pred_mean, y_pred_stddev, y_latent_mean, y_latent_stddev = (
                            model([current_x_test_image, current_x_test_label,
                                   tf.random.normal((1,) + latent_shape)], training=False))

                        y_pred_means.append(y_pred_mean)
                        y_pred_stddevs.append(y_pred_stddev)

                    y_pred_mean = tf.math.reduce_mean(y_pred_means, axis=0)
                    y_pred_stddev = tf.math.reduce_mean(y_pred_stddevs, axis=0)

                    output_image(y_pred_mean[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                                 standard_scaler, "{0}/mean.png".format(current_test_output_path))

                    y_pred = []

                    for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                        y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                    y_pred = tf.math.reduce_mean(y_pred, axis=0)
                else:
                    if discrete_bool:
                        y_pred_mean, y_pred_stddev, y_latent_quantised, y_latent_discretised = (
                            model([current_x_test_image, current_x_test_label], training=False))

                        output_image(y_latent_quantised[0], None, None, standard_scaler,
                                     "{0}/latent_quantised.png".format(current_test_output_path))

                        output_image(y_pred_mean[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                                     standard_scaler, "{0}/mean.png".format(current_test_output_path))

                        y_pred = []

                        for k in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                            y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                        y_pred = tf.math.reduce_mean(y_pred, axis=0)
                    else:
                        y_pred_mean, y_pred_stddev, y_latent = model([current_x_test_image, current_x_test_label],
                                                                     training=False)

                        output_image(y_latent[0], None, None, standard_scaler,
                                     "{0}/latent.png".format(current_test_output_path))

                        output_image(y_pred_mean[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                                     standard_scaler, "{0}/mean.png".format(current_test_output_path))

                        y_pred = []

                        for j in range(int(tf.math.ceil(tf.math.reduce_max(y_pred_stddev) * 32.0))):
                            y_pred.append(tf.random.normal(y_pred_mean.shape, y_pred_mean, y_pred_stddev))

                        y_pred = tf.math.reduce_mean(y_pred, axis=0)
            else:
                if gaussian_latent_bool:
                    y_pred, y_latent_mean, y_latent_stddev = (
                        model([current_x_test_image, current_x_test_label, tf.random.normal((1,) + latent_shape)],
                              training=False))

                    output_image(y_latent_mean[0], None, None, standard_scaler,
                                 "{0}/latent_mean.png".format(current_test_output_path))

                    output_image(y_latent_stddev[0], None, None, standard_scaler,
                                 "{0}/latent_stddev.png".format(current_test_output_path))

                    y_preds = [y_pred]

                    for k in range(int(tf.math.reduce_max([tf.math.ceil(tf.math.reduce_max(y_latent_stddev) * 32.0),
                                                           32.0])) - 1):
                        y_pred, y_latent_mean, y_latent_stddev = (
                            model([current_x_test_image, current_x_test_label,
                                   tf.random.normal((1,) + latent_shape)], training=False))

                        output_image(y_latent_mean[0], None, None, standard_scaler,
                                     "{0}/latent_mean.png".format(current_test_output_path))

                        output_image(y_latent_stddev[0], None, None, standard_scaler,
                                     "{0}/latent_stddev.png".format(current_test_output_path))

                        y_preds.append(y_pred)

                    y_pred = tf.math.reduce_mean(y_preds, axis=0)
                else:
                    if discrete_bool:
                        y_pred, y_latent_quantised, y_latent_discretised = (
                            model([current_x_test_image, current_x_test_label], training=False))

                        output_image(y_latent_quantised[0], None, None, standard_scaler,
                                     "{0}/latent_quantised.png".format(current_test_output_path))
                    else:
                        y_pred, y_latent = model([current_x_test_image, current_x_test_label], training=False)

                        output_image(y_latent[0], None, None, standard_scaler,
                                     "{0}/latent.png".format(current_test_output_path))
        else:
            y_pred, y_latent = model([current_x_test_image], training=False)

            output_image(y_latent[0], None, None, standard_scaler,
                         "{0}/latent.png".format(current_test_output_path))

        output_image(y_pred[0], current_x_test_original_shape, current_x_test_padding_mask[0], standard_scaler,
                     "{0}/output.png".format(current_test_output_path))

        output_image(current_x_test_image[0], current_x_test_original_shape, current_x_test_padding_mask[0],
                     standard_scaler, "{0}/input.png".format(current_test_output_path))

    return True


def main():
    print("main")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mkdir_p(output_path)

    x_train_images, x_test_images, x_train_labels, x_test_labels = get_input()
    (x_train_images, x_train_original_shapes, x_train_padding_masks, x_test_images, x_test_original_shapes,
     x_test_padding_masks, standard_scaler, x_train_labels, x_test_labels, preprocessors) = (
        preprocess_input(x_train_images, x_test_images, x_train_labels, x_test_labels))

    if conv_bool:
        if alex_bool:
            if gaussian_negative_log_likelihood_bool:
                if gaussian_latent_bool:
                    model = get_model_conv_alex_gaussian_latent_gaussian_negative_log_likelihood(x_train_images,
                                                                                                 x_train_labels)
                else:
                    if discrete_bool:
                        model = get_model_conv_alex_discrete_gaussian_negative_log_likelihood(x_train_images,
                                                                                              x_train_labels)
                    else:
                        model = get_model_conv_alex_gaussian_negative_log_likelihood(x_train_images, x_train_labels)
            else:
                if gaussian_latent_bool:
                    model = get_model_conv_alex_gaussian_latent(x_train_images, x_train_labels)
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
                                         use_ema=use_ema,
                                         ema_overwrite_frequency=ema_overwrite_frequency)

    batch_sizes, batch_sizes_epochs = get_batch_sizes(x_train_images)

    if gradient_accumulation_bool:
        model = train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images,
                                            x_train_original_shapes, x_train_padding_masks, x_test_images,
                                            x_test_original_shapes, x_test_padding_masks, standard_scaler,
                                            x_train_labels, x_test_labels)
    else:
        model = train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images, x_train_original_shapes,
                      x_train_padding_masks, x_test_images, x_test_original_shapes, x_test_padding_masks,
                      standard_scaler, x_train_labels, x_test_labels)

    if use_ema and epochs > 0:
        optimiser.finalize_variable_values(model.trainable_variables)

    test(model, x_test_images, x_test_original_shapes, x_test_padding_masks, standard_scaler, x_test_labels,
         "test")

    test(model, x_train_images, x_train_original_shapes, x_train_padding_masks, standard_scaler, x_train_labels,
         "train")

    return True


if __name__ == "__main__":
    main()
