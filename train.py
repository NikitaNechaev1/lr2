"""This module implements data feeding and training loop to create model
to classify X-Ray chest images as a lab example for BSU students.
"""


import argparse
import glob
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from datetime import datetime
from pathlib import Path
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
import os
import sys


LOG_DIR = 'logs'
SHUFFLE_BUFFER = 1
# BATCH_SIZE = 256
BATCH_SIZE = 4
# NUM_CLASSES = 6
NUM_CLASSES = 1
PARALLEL_CALLS=2
RESIZE_TO = 224
# TRAINSET_SIZE = 14034
TRAINSET_SIZE = 13
# VALSET_SIZE = 3000
VALSET_SIZE = 3


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def visualize_images(epoch, model, dataset, writer):
    item = iter(dataset).next()

    l_channel = item[:, :, :, 0]

    target_ab = item[:, :, :, 1:]
    target_image = np.zeros(item.shape)
    target_image[:, :, :, 0] = l_channel
    target_image[:, :, :, 1:] = target_ab

    predicted_ab = model(np.reshape(l_channel, (-1, 224, 224, 1)))
    predicted_image = np.zeros(item.shape)
    predicted_image[:, :, :, 0] = l_channel
    predicted_image[:, :, :, 1:] = predicted_ab

    # target_rgb = tfio.experimental.color.lab_to_rgb(target_image)
    # predicted_rgb = tfio.experimental.color.lab_to_rgb(predicted_image)

    with writer.as_default():
        tf.summary.image('Target Lab', np.reshape(target_image, (-1, 224, 224, 3)), step=epoch),
        tf.summary.image('Result Lab', np.reshape(predicted_image, (-1, 224, 224, 3)), step=epoch),
        tf.summary.image('Target L channel', np.reshape(l_channel,(-1, 224, 224, 1)), step=epoch),
        tf.summary.image('Target ab channels', np.reshape(target_ab, (-1, 224, 224, 2)), step=epoch)

def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')#,
        # 'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    example = tf.io.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image']#, tf.one_hot(example['image/class/label'], depth=NUM_CLASSES)


def normalize(image, label):
    return tf.image.per_image_standardization(image), label

# def resize(image, label):
#     return tf.image.resize(image, tf.constant([RESIZE_TO, RESIZE_TO])), label

def resize(image):
    return tf.image.resize(image, tf.constant([RESIZE_TO, RESIZE_TO]))


def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    # """
    # return tf.data.TFRecordDataset(filenames)\
    #     .map(parse_proto_example)\
    #     .map(resize)\
    #     .map(normalize)\
    #     .batch(batch_size)\
    #     .prefetch(batch_size)

    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .batch(batch_size)\
        .prefetch(batch_size)


def build_model():
    # return tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=(224,224,3)),
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
    #     tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2,2), activation='relu', padding='same'),
    #     tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), activation='relu', padding='same'),
    #     tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, strides=(2,2), activation='relu', padding='same')
    # ])
    model = tf.keras.models.Sequential()

    # model.add(RandomRotation(factor=0.45))
    # model.add(RandomFlip(mode='horizontal'))

    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 1)))
    model.add(tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2DTranspose(2, (3, 3), strides=2, activation='relu', padding='same'))
    return model


def main():
    # print(sys.argv[0])
    # print(os.getcwd())
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.chdir(os.path.dirname(sys.argv[0])) # change working directory as script directory
    args = argparse.ArgumentParser()
    args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files')
    args.add_argument('--test', type=str, help='Glob pattern to collect test tfrecord files')
    args = args.parse_args()

    current_dir = os.getcwd()

    train_dir = Path(current_dir + '/' + args.train)
    file_list_train = [str(pp) for pp in train_dir.glob("*")]

    valid_dir = Path(current_dir + '/' + args.test)
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]
    
    train_dataset = create_dataset(file_list_train, BATCH_SIZE)
    validation_dataset = create_dataset(file_list_valid, BATCH_SIZE)

    # train_dataset = create_dataset(glob.glob(train_path), BATCH_SIZE)
    # validation_dataset = create_dataset(glob.glob(test_path), BATCH_SIZE)
    [print(np.shape(i)) for i in validation_dataset.as_numpy_iterator()]
    
    validation_y = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in validation_dataset.as_numpy_iterator()]

    x = [np.reshape(i[:, :, :, 0], (-1, 224, 224, 1)) for i in train_dataset.as_numpy_iterator()]
    y = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in train_dataset.as_numpy_iterator()]

    model = build_model()

    model.compile(
        optimizer=tf.optimizers.SGD(lr=0.01, momentum=0.9),
        # loss=tf.keras.losses.categorical_crossentropy
        loss=tf.keras.losses.mean_squared_error
        # metrics=[tf.keras.metrics.categorical_accuracy],
    )

    log_dir='{}/ilcd-{}'.format(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    file_writer = tf.summary.create_file_writer(log_dir)
    
    model.fit(
        # train_dataset,
        # epochs=200,
        # validation_data=validation_dataset,
        
        # callbacks=[
        #     tf.keras.callbacks.TensorBoard(log_dir),
        # ]
        x=x.pop(),
        y=y.pop(),
        epochs=100,
        validation_data=validation_y.pop().all(),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images(epoch, model, validation_dataset, file_writer)
            )
        ]
    )
    
    print(model.summary())


if __name__ == '__main__':
    main()
