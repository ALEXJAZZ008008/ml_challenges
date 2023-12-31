# Copyright University College London 2023
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import einops
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle


dataset_name = "cifar10"
output_path = "../output/classification/"

read_data_from_storage_bool = False

preprocess_list_bool = False
convert_rgb_to_greyscale_bool = False
max_output_dimension_size = None

conv_bool = True
deep_bool = True
alex_bool = True
dense_layers = 4
filters = [64, 128, 256, 512, 1024]
conv_layers = [2, 2, 2, 2, 2]
num_heads = 4
key_dim = filters[-1]

learning_rate = 1e-04
weight_decay = 0.0

gradient_accumulation_bool = False

epochs = 1
min_batch_size = 32

if gradient_accumulation_bool:
    max_batch_size = None
else:
    max_batch_size = 32

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

    return x_train, x_test, y_train, y_test


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

    if max_output_dimension_size is not None:
        if output_dimension_size > max_output_dimension_size:
            output_dimension_size = max_output_dimension_size

    for i in range(images_len):
        image = get_data_from_storage(images[i])

        image = rescale(image, output_dimension_size / np.max(image.shape[:-1]), mode="constant", clip=False,
                        preserve_range=True, channel_axis=-1)

        images[i] = set_data_from_storage(image, images[i])

    return images


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


def normalise_images_list(x_train, x_test):
    print("normalise_images_list")

    standard_scaler = StandardScaler()

    x_train_len = len(x_train)

    for i in range(x_train_len):
        current_x_train = get_data_from_storage(x_train[i])

        standard_scaler.partial_fit(np.reshape(current_x_train, (-1, 1)))

    for i in range(x_train_len):
        current_x_train = get_data_from_storage(x_train[i])

        current_x_train = np.reshape(standard_scaler.transform(np.reshape(current_x_train, (-1, 1))),
                                     current_x_train.shape)

        x_train[i] = set_data_from_storage(current_x_train, x_train[i])

    for i in range(len(x_test)):
        current_x_test = get_data_from_storage(x_test[i])

        current_x_test = np.reshape(standard_scaler.transform(np.reshape(current_x_test, (-1, 1))),
                                    current_x_test.shape)

        x_test[i] = set_data_from_storage(current_x_test, x_test[i])

    return x_train, x_test


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
    print("convert_images_to_tensor_list")

    for i in range(len(images)):
        image = get_data_from_storage(images[i])

        image = tf.convert_to_tensor(image)

        images[i] = set_data_from_storage(image, images[i])

    return images


def preprocess_images_list(x_train, x_test):
    print("preprocess_images")

    x_train = rescale_images_list(x_train)
    x_test = rescale_images_list(x_test)

    x_train, x_test = normalise_images_list(x_train, x_test)

    x_train = pad_images_list(x_train)
    x_test = pad_images_list(x_test)

    x_train = convert_images_to_tensor_list(x_train)
    x_test = convert_images_to_tensor_list(x_test)

    return x_train, x_test


def rescale_images_array(images):
    print("rescale_images")

    max_dimension_size = np.max(images.shape[1:-1])
    output_dimension_size = get_next_geometric_value(max_dimension_size, 2.0)

    if max_output_dimension_size is not None:
        if output_dimension_size > max_output_dimension_size:
            output_dimension_size = max_output_dimension_size

    rescaled_images = []

    for i in range(len(images)):
        rescaled_images.append(rescale(images[i], output_dimension_size / max_dimension_size, mode="constant",
                                       clip=False, preserve_range=True, channel_axis=-1))

    images = np.array(rescaled_images)

    return images


def normalise_images_array(x_train, x_test):
    print("normalise_images_array")

    standard_scaler = StandardScaler()

    x_train = np.reshape(standard_scaler.fit_transform(np.reshape(x_train, (-1, 1))), x_train.shape)
    x_test = np.reshape(standard_scaler.transform(np.reshape(x_test, (-1, 1))), x_test.shape)

    return x_train, x_test


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


def preprocess_images_array(x_train, x_test):
    print("preprocess_images")

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = rescale_images_array(x_train)
    x_test = rescale_images_array(x_test)

    x_train, x_test = normalise_images_array(x_train, x_test)

    x_train = pad_images_array(x_train)
    x_test = pad_images_array(x_test)

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

    return x_train, x_test


def preprocess_labels(y_train, y_test):
    print("preprocess_labels")

    one_hot_encoder = OneHotEncoder(sparse_output=False)

    y_train = np.reshape(one_hot_encoder.fit_transform(np.reshape(y_train, (-1, 1))),
                         (y_train.shape[0], -1))
    y_test = np.reshape(one_hot_encoder.transform(np.reshape(y_test, (-1, 1))),
                        (y_test.shape[0], -1))

    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    return y_train, y_test


def preprocess_input(x_train, x_test, y_train, y_test):
    print("preprocess_input")

    if convert_rgb_to_greyscale_bool:
        x_train = convert_rgb_to_greyscale(x_train)
        x_test = convert_rgb_to_greyscale(x_test)

    if read_data_from_storage_bool or preprocess_list_bool:
        x_train, x_test = preprocess_images_list(x_train, x_test)
    else:
        x_train, x_test = preprocess_images_array(x_train, x_test)

    y_train, y_test = preprocess_labels(y_train, y_test)

    return x_train, x_test, y_train, y_test


def get_model_dense(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = current_x_train_images.shape
    else:
        input_shape = x_train.shape[1:]

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


def get_previous_geometric_value(an, a0):
    n = np.log2(an / a0) + 1

    if not n.is_integer():
        an = a0 * np.power(2.0, (np.floor(n) - 1.0))

    return an


def get_model_deep_dense(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = current_x_train_images.shape
    else:
        input_shape = x_train.shape[1:]

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


def get_model_conv(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = current_x_train_images.shape
    else:
        input_shape = x_train.shape[1:]

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

    for j in range(conv_layers[-1]):
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


def get_model_conv_alex(x_train, y_train):
    print("get_model")

    if read_data_from_storage_bool:
        current_x_train_images = get_data_from_storage(x_train[0])

        input_shape = current_x_train_images.shape
    else:
        input_shape = x_train.shape[1:]

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
            x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

        x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding="same",
                                       kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
        x = tf.keras.layers.Add()([x, x_res])

        x = tf.keras.layers.Lambda(einops.rearrange,
                                   arguments={"pattern": "b (h h1) (w w1) c -> b h w (c h1 w1)", "h1": 2, "w1": 2})(x)

    x_res = x

    for i in range(conv_layers[-1]):
        x = tf.keras.layers.Conv2D(filters=filters[-1],
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x_res = tf.keras.layers.Conv2D(filters=x.shape[-1],
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.orthogonal)(x_res)
    x = tf.keras.layers.Add()([x, x_res])

    if num_heads is not None and key_dim is not None:
        x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                               key_dim=key_dim,
                                               kernel_initializer=tf.keras.initializers.orthogonal)(x, x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(units=get_next_geometric_value(x.shape[-1], 2.0),
                              kernel_initializer=tf.keras.initializers.he_uniform)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    for i in range(dense_layers - 1):
        x = tf.keras.layers.Dense(units=get_previous_geometric_value(x.shape[-1] - 1, 2.0),
                                  kernel_initializer=tf.keras.initializers.he_uniform)(x)
        x = tf.keras.layers.Lambda(tf.keras.activations.swish)(x)

    x = tf.keras.layers.Dense(units=y_train.shape[-1],
                              kernel_initializer=tf.keras.initializers.he_uniform)(x)
    x = tf.keras.layers.Lambda(tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=[x_input],
                           outputs=[x])

    return model


def sinespace(start, stop, num):
    linspaced = np.linspace(0.0, 90.0, num)

    sinspaced = np.sin(np.deg2rad(linspaced))

    sinspaced_min = np.min(sinspaced)
    sinspaced = start + ((sinspaced - sinspaced_min) * ((stop - start) / ((np.max(sinspaced) - sinspaced_min) +
                                                                          np.finfo(np.float32).eps)))

    return sinspaced


def get_batch_sizes(x_train):
    print("get_batch_sizes")

    current_min_batch_size = get_next_geometric_value(min_batch_size, 2.0)

    x_train_len = len(x_train)

    if max_batch_size is not None:
        if max_batch_size < current_min_batch_size or max_batch_size > x_train_len:
            current_max_batch_size = x_train_len
        else:
            current_max_batch_size = max_batch_size
    else:
        current_max_batch_size = x_train_len

    batch_sizes = [current_min_batch_size]

    while True:
        current_batch_size = int(np.round(batch_sizes[-1] * 2.0))

        if current_batch_size <= current_max_batch_size:
            batch_sizes.append(current_batch_size)
        else:
            break

    if current_max_batch_size >= x_train_len > batch_sizes[-1]:
        batch_sizes.append(x_train_len)

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
        image = rescale(image, random.uniform(min_scale, max_scale), mode="constant", clip=False, preserve_range=True,
                        channel_axis=-1)

    if rotate_bool:
        image = scipy.ndimage.rotate(image, angle=random.uniform(min_angle, max_angle), axes=(0, 1), order=1)

    if translate_bool:
        image = translate_image(image)

    image = pad_image(image, input_dimension_size)
    image = crop_image(image, input_dimension_size)

    image = tf.convert_to_tensor(image)

    return image


def get_loss(y_true, y_pred):
    loss = tf.math.reduce_mean(tf.keras.metrics.categorical_crossentropy(y_true, y_pred))

    return loss


def get_accuracy(y_true, y_pred):
    accuracy = tf.math.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred)) * 100.0

    return accuracy


def train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train, y_train):
    print("train")

    y_train = tf.expand_dims(y_train, axis=1)

    x_train_len = len(x_train)

    current_batch_size = None
    indices = list(range(x_train_len))

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    for i in range(epochs):
        for j in range(batch_sizes_epochs_len - 1, 0, -1):
            if batch_sizes_epochs[j - 1] <= i < batch_sizes_epochs[j]:
                current_batch_size = batch_sizes[j - 1]

                break

        iterations = int(np.floor(x_train_len / current_batch_size))

        random.shuffle(indices)

        for j in range(iterations):
            accumulated_gradients = [tf.zeros_like(trainable_variable) for trainable_variable in
                                     model.trainable_variables]

            current_index = current_batch_size * j

            losses = []
            accuracies = []

            for m in range(current_batch_size):
                current_x_train = get_data_from_storage(x_train[current_index])

                current_x_train = augmentation(current_x_train)
                y_true = y_train[current_index]

                current_x_train = tf.expand_dims(current_x_train, axis=0)

                with tf.GradientTape() as tape:
                    y_pred = model([current_x_train], training=True)

                    loss = get_loss(y_true, y_pred)

                gradients = tape.gradient(loss, model.trainable_weights)

                accumulated_gradients = [(accumulated_gradient + gradient) for accumulated_gradient, gradient in
                                         zip(accumulated_gradients, gradients)]

                losses.append(loss)
                accuracies.append(get_accuracy(y_true, y_pred))

                current_index = current_index + 1

            gradients = [gradient / current_batch_size for gradient in accumulated_gradients]
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            loss = tf.math.reduce_mean(losses)
            accuracy = tf.math.reduce_mean(accuracies)

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss: {5:12} Accuracy: {6:6}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), str(loss.numpy()),
                str(accuracy.numpy())))

    return model


def train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train, y_train):
    print("train")

    x_train_len = len(x_train)

    current_batch_size = None
    indices = list(range(x_train_len))

    batch_sizes_epochs_len = len(batch_sizes_epochs)

    for i in range(epochs):
        for j in range(batch_sizes_epochs_len - 1, 0, -1):
            if batch_sizes_epochs[j - 1] <= i <= batch_sizes_epochs[j]:
                current_batch_size = batch_sizes[j - 1]

                break

        iterations = int(np.floor(x_train_len / current_batch_size))

        random.shuffle(indices)

        for j in range(iterations):
            current_index = current_batch_size * j

            current_x_train_list = []
            y_true = []

            for k in range(current_batch_size):
                current_x_train = get_data_from_storage(x_train[current_index])

                current_x_train_list.append(augmentation(current_x_train))
                y_true.append(y_train[current_index])

                current_index = current_index + 1

            current_x_train_list = tf.convert_to_tensor(current_x_train_list)
            y_true = tf.convert_to_tensor(y_true)

            with tf.GradientTape() as tape:
                y_pred = model([current_x_train_list], training=True)

                loss = get_loss(y_true, y_pred)

            gradients = tape.gradient(loss, model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimiser.apply_gradients(zip(gradients, model.trainable_weights))

            accuracy = get_accuracy(y_true, y_pred)

            print("Epoch: {0:3}/{1:3} Batch size: {2:6} Iteration: {3:6}/{4:6} Loss: {5:12} Accuracy: {6:6}%".format(
                str(i + 1), str(epochs), str(current_batch_size), str(j + 1), str(iterations), str(loss.numpy()),
                str(accuracy.numpy())))

    return model


def test(model, x_test, y_test):
    print("test")

    losses = []
    accuracies = []

    x_test_len = len(x_test)

    for i in range(int(np.floor(x_test_len / test_batch_size))):
        current_x_test_list = []
        y_true = []

        for j in range(test_batch_size):
            current_index = (i * test_batch_size) + j

            current_x_test = get_data_from_storage(x_test[current_index])

            current_x_test_list.append(current_x_test)
            y_true.append(y_test[current_index])

        current_x_test_list = tf.convert_to_tensor(current_x_test_list)
        y_true = tf.convert_to_tensor(y_true)

        y_pred = model([current_x_test_list], training=True)

        losses.append(get_loss(y_true, y_pred))
        accuracies.append(get_accuracy(y_true, y_pred))

    loss = tf.math.reduce_mean(losses)
    accuracy = tf.math.reduce_mean(accuracies)

    print("Loss: {0:12} Accuracy: {1:6}%".format(str(loss.numpy()), str(accuracy.numpy())))

    return model


def main():
    print("main")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if read_data_from_storage_bool:
        mkdir_p(output_path)

    x_train, x_test, y_train, y_test = get_input()
    x_train, x_test, y_train, y_test = preprocess_input(x_train, x_test, y_train, y_test)

    if conv_bool:
        if alex_bool:
            model = get_model_conv_alex(x_train, y_train)
        else:
            model = get_model_conv(x_train, y_train)
    else:
        if deep_bool:
            model = get_model_deep_dense(x_train, y_train)
        else:
            model = get_model_dense(x_train, y_train)

    model.summary()

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         weight_decay=weight_decay,
                                         use_ema=True)

    batch_sizes, batch_sizes_epochs = get_batch_sizes(x_train)

    if gradient_accumulation_bool:
        model = train_gradient_accumulation(model, optimiser, batch_sizes, batch_sizes_epochs, x_train, y_train)
    else:
        model = train(model, optimiser, batch_sizes, batch_sizes_epochs, x_train, y_train)

    if epochs > 0:
        optimiser.finalize_variable_values(model.trainable_variables)

    test(model, x_test, y_test)

    return True


if __name__ == "__main__":
    main()
