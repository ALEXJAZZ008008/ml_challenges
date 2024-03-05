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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import einops
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm
import pickle


np.seterr(all="print")


dataset_name = "cifar10"
output_path = "../output/classification/"

read_data_from_storage_bool = False

preprocess_list_bool = False
greyscale_bool = True
output_dimension_size = 32

deep_bool = True
conv_bool = True
alex_bool = True
dense_layers = 4
filters = [32, 64, 128, 256, 512, 1024]
conv_layers = [2, 2, 2, 2, 2, 2]
num_heads = 4
key_dim = 32

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

epochs = 10

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
gaussian_bool = True
max_sigma = 1.0
sharpen_bool = True
max_radius = 1.0
max_amount = 1.0
scale_bool = True
min_scale = 0.75
max_scale = 1.25
rotate_bool = True
min_angle = -45.0
max_angle = 45.0
translate_bool = True
translate_proportion = 0.25

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

    x_train_output_path = "{0}/x_train/".format(output_path)
    x_test_output_path = "{0}/x_test/".format(output_path)

    if read_data_from_storage_bool:
        mkdir_p(x_train_output_path)
        mkdir_p(x_test_output_path)

    dataset = tfds.load(dataset_name)

    dataset_train = dataset["train"]
    dataset_test = dataset["test"]
    # dataset_test = dataset["validation"]

    x_train = []
    x_test = []

    y_train = []
    y_test = []

    for i, example in enumerate(dataset_train):
        if read_data_from_storage_bool:
            current_x_train = example["image"].numpy().astype(np.float32)
            y_train.append(example["label"].numpy().astype(np.float32))

            x_train.append("{0}/{1}.pkl".format(x_train_output_path, str(i)))

            with open(x_train[-1], "wb") as file:
                pickle.dump(current_x_train, file)
        else:
            x_train.append(example["image"].numpy().astype(np.float32))
            y_train.append(example["label"].numpy().astype(np.float32))

    for i, example in enumerate(dataset_test):
        if read_data_from_storage_bool:
            current_x_test = example["image"].numpy().astype(np.float32)
            y_test.append(example["label"].numpy().astype(np.float32))

            x_test.append("{0}/{1}.pkl".format(x_test_output_path, str(i)))

            with open(x_test[-1], "wb") as file:
                pickle.dump(current_x_test, file)
        else:
            x_test.append(example["image"].numpy().astype(np.float32))
            y_test.append(example["label"].numpy().astype(np.float32))

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    return x_train, x_test, y_train, y_test


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


def preprocess_labels(y_train, y_test):
    print("preprocess_labels")

    one_hot_encoder = OneHotEncoder(sparse_output=False)

    x_train_labels_preprocessed = np.reshape(one_hot_encoder.fit_transform(np.reshape(y_train, (-1, 1))),
                                             (y_train.shape[0], -1))
    x_test_labels_preprocessed = np.reshape(one_hot_encoder.transform(np.reshape(y_test, (-1, 1))),
                                            (y_test.shape[0], -1))

    x_train_labels_preprocessed = tf.convert_to_tensor(x_train_labels_preprocessed)
    x_test_labels_preprocessed = tf.convert_to_tensor(x_test_labels_preprocessed)

    return x_train_labels_preprocessed, x_test_labels_preprocessed


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

    x_train_labels_preprocessed, x_test_labels_preprocessed = preprocess_labels(x_train_labels, x_test_labels)

    return (x_train_images_preprocessed, x_train_original_shapes, x_train_padding_masks, x_test_images_preprocessed,
            x_test_original_shapes, x_test_padding_masks, standard_scaler, x_train_labels_preprocessed,
            x_test_labels_preprocessed)


def get_model_dense(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1], 2.0))(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[-1])(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_previous_geometric_value(value, base):
    power = np.log2(value / base) + 1

    if not power.is_integer():
        previous_value = base * np.power(2.0, (np.floor(power) - 1.0))
    else:
        previous_value = copy.deepcopy(value)

    return previous_value


def get_model_deep_dense(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1], 2.0))(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(dense_layers - 1):
        x = tf.keras.layers.Dense(units=get_previous_geometric_value(x.shape[-1] - 1, 2.0))(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[-1])(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_model_deep_conv(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    for i in range(len(filters)):
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
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1], 2.0))(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[-1])(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_model_conv(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = x_input

    for i in range(len(filters)):
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
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same")(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.relu)(x)

    x = tf.keras.layers.Conv2D(filters=y_train.shape[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def get_model_conv_alex(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = list(current_x_train_images.shape)
    else:
        input_shape = list(x_train.shape[1:])

    x_input = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(filters=filters[0],
                               kernel_size=(7, 7),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=tf.keras.initializers.orthogonal)(x_input)

    filters_len = len(filters)

    for i in range(filters_len - 1):
        x_res = x

        for j in range(conv_layers[i]):
            x = tf.keras.layers.Conv2D(filters=filters[i],
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x)
            x = tf.keras.layers.GroupNormalization(groups=8)(x)
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
        x = tf.keras.layers.GroupNormalization(groups=8)(x)
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

    x = tf.keras.layers.Conv2D(filters=x.shape[-1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same")(x)
    x = tf.keras.layers.GlobalMaxPool2D()(x)

    x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1], 2.0),
                              kernel_initializer=tf.keras.initializers.orthogonal)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    for i in range(dense_layers - 1):
        x = tf.keras.layers.Dense(units=get_previous_geometric_value(x.shape[-1] - 1, 2.0),
                                  kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[-1],
                              kernel_initializer=tf.keras.initializers.orthogonal)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

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
                                           amount=random.uniform(0.0, max_amount), preserve_range=True, channel_axis=-1)

        if scale_bool:
            augmented_image = rescale(augmented_image, random.uniform(min_scale, max_scale), order=3, preserve_range=True,
                                      channel_axis=-1)

        if rotate_bool:
            augmented_image = scipy.ndimage.rotate(augmented_image, angle=random.uniform(min_angle, max_angle), axes=(0, 1),
                                                   order=1)

        if translate_bool:
            augmented_image = translate_image(augmented_image)

        augmented_image = pad_image(augmented_image, None, original_shape)
        augmented_image = crop_image(augmented_image, None, original_shape)

        augmented_image = pad_image(augmented_image, None, input_shape)

    augmented_image = tf.convert_to_tensor(augmented_image)

    return augmented_image


def get_loss(y_true, y_pred):
    loss = tf.math.reduce_mean(tf.keras.metrics.categorical_crossentropy(y_true, y_pred))

    return loss


def get_accuracy(y_true, y_pred):
    accuracy = tf.math.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred)) * 100.0

    return accuracy


def train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images,
                                x_train_original_shapes, x_train_padding_masks, x_train_labels):
    print("train")

    x_train_labels = tf.expand_dims(x_train_labels, axis=1)

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
            current_gradient_accumulation_batch_size = (
                int(np.floor(current_batch_size / gradient_accumulation_batch_size)))

            losses = []
            accuracies = []

            for k in range(current_gradient_accumulation_batch_size):
                current_x_train_images = []
                current_x_train_labels = []

                for n in range(gradient_accumulation_batch_size):
                    current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])
                    current_x_train_original_shape = x_train_original_shapes[indices[current_index]]
                    current_x_train_padding_mask = get_data_from_storage(x_train_padding_masks[indices[current_index]])

                    current_x_train_images.append(augmentation(current_x_train_image, current_x_train_original_shape,
                                                               current_x_train_padding_mask))
                    current_x_train_labels.append(x_train_labels[indices[current_index]])

                    current_index = current_index + 1

                current_x_train_images = tf.convert_to_tensor(current_x_train_images)
                current_x_train_labels = tf.convert_to_tensor(current_x_train_labels)

                with tf.GradientTape() as tape:
                    y_pred = model([current_x_train_images], training=True)

                    loss = tf.math.reduce_sum([get_loss(current_x_train_labels, y_pred),
                                               tf.math.reduce_sum(model.losses)])

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)
                accuracies.append(get_accuracy(current_x_train_labels, y_pred))

            gradients = [gradient / current_batch_size for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            accuracy = tf.math.reduce_mean(accuracies)

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss: {5:18} Accuracy: {6:18}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), str(loss.numpy()),
                str(accuracy.numpy())))

        if use_ema and ema_overwrite_frequency is None:
            optimiser.finalize_variable_values(model.trainable_variables)

    return model


def train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images, x_train_original_shapes,
          x_train_padding_masks, x_train_labels):
    print("train")

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
            current_x_train_labels = []

            for k in range(current_batch_size):
                current_x_train_image = get_data_from_storage(x_train_images[indices[current_index]])
                current_x_train_original_shape = x_train_original_shapes[indices[current_index]]
                current_x_train_padding_mask = get_data_from_storage(x_train_padding_masks[indices[current_index]])

                current_x_train_images.append(augmentation(current_x_train_image, current_x_train_original_shape,
                                                           current_x_train_padding_mask))
                current_x_train_labels.append(x_train_labels[indices[current_index]])

                current_index = current_index + 1

            current_x_train_images = tf.convert_to_tensor(current_x_train_images)
            current_x_train_labels = tf.convert_to_tensor(current_x_train_labels)

            with tf.GradientTape() as tape:
                y_pred = model([current_x_train_images], training=True)

                loss = tf.math.reduce_sum([get_loss(current_x_train_labels, y_pred),
                                           tf.math.reduce_sum(model.losses)])

            gradients = tape.gradient(loss, model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            accuracy = get_accuracy(current_x_train_labels, y_pred)

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss: {5:18} Accuracy: {6:18}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), str(loss.numpy()),
                str(accuracy.numpy())))

        if use_ema and ema_overwrite_frequency is None:
            optimiser.finalize_variable_values(model.trainable_variables)

    return model


def test(model, x_test_images, x_test_labels):
    print("test")

    losses = []
    accuracies = []

    x_test_images_len = len(x_test_images)

    for i in range(int(np.floor(x_test_images_len / test_batch_size))):
        current_x_test_images = []
        y_true = []

        for j in range(test_batch_size):
            current_index = (i * test_batch_size) + j

            current_x_test_image = get_data_from_storage(x_test_images[current_index])

            current_x_test_images.append(current_x_test_image)
            y_true.append(x_test_labels[current_index])

        current_x_test_images = tf.convert_to_tensor(current_x_test_images)
        y_true = tf.convert_to_tensor(y_true)

        y_pred = model([current_x_test_images], training=True)

        losses.append(get_loss(y_true, y_pred))
        accuracies.append(get_accuracy(y_true, y_pred))

    loss = tf.math.reduce_mean(losses)
    accuracy = tf.math.reduce_mean(accuracies)

    print("Loss: {0:18} Accuracy: {1:18}%".format(str(loss.numpy()), str(accuracy.numpy())))

    return model


def main():
    print("main")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if read_data_from_storage_bool:
        mkdir_p(output_path)

    x_train_images, x_test_images, x_train_labels, x_test_labels = get_input()
    (x_train_images, x_train_original_shapes, x_train_padding_masks, x_test_images, x_test_original_shapes,
     x_test_padding_masks, standard_scaler, x_train_labels, x_test_labels) = (
        preprocess_input(x_train_images, x_test_images, x_train_labels, x_test_labels))

    if conv_bool:
        if alex_bool:
            model = get_model_conv_alex(x_train_images, x_train_labels)
        else:
            if deep_bool:
                model = get_model_deep_conv(x_train_images, x_train_labels)
            else:
                model = get_model_conv(x_train_images, x_train_labels)
    else:
        if deep_bool:
            model = get_model_deep_dense(x_train_images, x_train_labels)
        else:
            model = get_model_dense(x_train_images, x_train_labels)

    model.summary()

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         weight_decay=weight_decay,
                                         use_ema=use_ema,
                                         ema_overwrite_frequency=ema_overwrite_frequency)

    batch_sizes, batch_sizes_epochs = get_batch_sizes(x_train_images)

    if gradient_accumulation_bool:
        model = train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images,
                                            x_train_original_shapes, x_train_padding_masks, x_train_labels)
    else:
        model = train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train_images, x_train_original_shapes,
                      x_train_padding_masks, x_train_labels)

    if use_ema and epochs > 0:
        optimiser.finalize_variable_values(model.trainable_variables)

    test(model, x_test_images, x_test_labels)

    return True


if __name__ == "__main__":
    main()
