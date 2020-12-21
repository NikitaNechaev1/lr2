#!/usr/bin/python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements TFRecords creation based on filetree and train/val ratio as a lab example for BSU students.
The image data set is expected to reside in JPEG files located in the following directory structure:
  data_dir/image0.jpeg
  data_dir/image1.jpg
  ...


This TensorFlow script converts the training and evaluation data into a sharded data set consisting of TFRecord files
  output_directory/train-00000-of-01024
  ...
  output_directory/train-01023-of-01024
and
  output_directory/validation-00000-of-00128
  ...
  output_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set.
The labels file contains a list of valid labels where each line corresponds to a label.
We map each label contained in the file to an integer corresponding to the line number starting from 0.

Each record within the TFRecord file is a serialized
Example proto. The Example proto contains many fields, the most important are:
  image/encoded: string containing JPEG encoded image in RGB colorspace
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
tf.compat.v1.disable_eager_execution()

flags.DEFINE_string('input', default=None, help='Data directory')
flags.DEFINE_string('output', default=None, help='Output directory')
flags.DEFINE_integer('shards', 5, 'Number of shards per split of TFRecord files.')
flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')
flags.DEFINE_string('labels_file', 'labels', 'Labels file')

FLAGS = flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width):
    """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.compat.v1.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.compat.v1.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        # self._decode_jpeg = tfio.experimental.color.rgb_to_lab(self._decode_jpeg_data)
        
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image_jpeg = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        image_jpeg = tf.cast(image_jpeg, tf.float16)
        image = tfio.experimental.color.rgb_to_lab(image_jpeg)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _is_png(filename):
    """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
    return filename.endswith('.png')


def _process_image(filename, coder):
    """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)


    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output, output_filename)
        writer = tf.io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image_buffer, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
    """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.

        data_dir/another-image.JPEG
        data_dir/my-image.jpg

    Returns:
      filenames: list of strings; each string is a path to an image file.
  """
    print('Determining list of input files from %s.' % data_dir)


    filenames = []

    # Construct the list of JPEG files and labels.
    # for text in unique_labels:
    # jpeg_file_path = '%s/%s/*' % (data_dir, text)
    jpeg_file_path = data_dir + '/*'
    matching_files = tf.io.gfile.glob(jpeg_file_path)

    filenames.extend(matching_files)

    print('Found %d JPEG files inside %s.' % (len(filenames), data_dir))
    return filenames

def main(_):
    assert FLAGS.input, ('Specify data root directory with --input flag')
    assert FLAGS.output, ('Specify destination directory with --output flag')
    assert not FLAGS.shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.shards')
    print('Saving results to %s' % FLAGS.output)

    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    split = 'lab'
    names = _find_image_files(FLAGS.input)
    # _process_image_files(split, names, len(names))
    _process_image_files(split, names, FLAGS.shards)


if __name__ == '__main__':
    app.run(main)
