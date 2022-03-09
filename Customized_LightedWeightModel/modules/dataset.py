import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization

def Annotation_loading(filename, mode):
    pkl = pickle.load(open(filename, 'rb'))
    if mode == 'train':
        return pkl['Training_Set']['path'], pkl['Training_Set']['label']
    elif mode == 'val':
        return pkl['Validation_Set']['path'], pkl['Validation_Set']['label']
    elif mode == 'test':
        return pkl['Testing_Set']['path'], pkl['Testing_Set']['label']

def _transform_images():
    def transform_images(x_train):
        seed = (random.randint(1, 9),random.randint(1, 9))
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        # x_train = tf.image.stateless_random_brightness(x_train, max_delta=0.1, seed=seed)
        # x_train = tf.image.stateless_random_contrast(x_train, lower=0.7, upper=1.3, seed=seed)
        # x_train /= 255.
        x_train = (x_train - 127.5) / 127.5
        return x_train
    return transform_images

def _transform_val_images():
    def transform_val_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        # x_train /= 255.
        x_train = (x_train - 127.5) / 127.5
        return x_train
    return transform_val_images


def image_loading(ImgPath, label):
    Img = tf.image.decode_jpeg(tf.io.read_file(ImgPath), channels=3)
    Img = _transform_images()(Img)
    label = tf.cast(label, tf.float32)
    return Img, label

def image_loading_val(ImgPath, label):
    Img = tf.image.decode_jpeg(tf.io.read_file(ImgPath), channels=3)
    Img = _transform_val_images()(Img)
    label = tf.cast(label, tf.float32)
    return Img, label


def load_dataset(pkl_files, batch_size, mode, buffer_size=100):
    Paths, Labels = Annotation_loading(pkl_files, mode)
    if mode=='train':
        dataset = (
            tf.data.Dataset.from_tensor_slices((Paths, Labels))
            .map(image_loading, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .repeat()
            .shuffle(buffer_size=buffer_size)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
    else:
        dataset = (
            tf.data.Dataset.from_tensor_slices((Paths, Labels))
            .map(image_loading_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
    return dataset, len(Labels)
