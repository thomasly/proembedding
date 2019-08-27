import sys
import os
import math
import logging
import argparse

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.layers import BatchNormalization
import numpy as np

from data import TOUGH_POINT


class PointNet(Model):

    def __init__(self, channels, classes=4):
        super(PointNet, self).__init__()
        conv_kernel = (1, channels)
        self.conv2d_input = Conv2D(64, conv_kernel, activation="relu")
        self.conv2d_64_64 = Conv2D(64, (1, 64), activation="relu")
        self.conv2d_64_128 = Conv2D(128, (1, 64), activation="relu")
        self.conv2d_128_1024 = Conv2D(1024, (1, 128), activation="relu")
        self.maxpool = MaxPool2D((1, 1024))
        self.dense_512 = Dense(512, activation="relu")
        self.dense_256 = Dense(256, activation="relu")
        self.dense_output = Dense(classes, activation="softmax")
        self.perm = (0, 1, 3, 2)

    def call(self, x):
        x = self.conv2d_input(x) # batch_size x pc_len x 1 x 64
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 64 x 1
        x = self.conv2d_64_64(x) # batch_size x pc_len x 1 x 64
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 64 x 1
        x = self.conv2d_64_64(x) # batch_size x pc_len x 1 x 64
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 64 x 1
        x = self.conv2d_64_128(x) # batch_size x pc_len x 1 x 128
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 128 x 1
        x = self.conv2d_128_1024(x) # batch_size x pc_len x 1 x 1024
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 1024 x 1
        x = self.maxpool(x) # batch_size x pc_len x 1 x 1
        x = self.dense_512(x)
        x = self.dense_256(x)
        x = self.dense_output(x)
        return x


def parse_argv(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resi-channel",
                        help="If to use residue channel in the input.",
                        action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Size of mini batch.")
    return parser.parse_args(argv)


def train():
    args = parse_argv(sys.argv[1:])
    resi_name_channel = args.resi_channel
    batch_size = args.batch_size
    tp = TOUGH_POINT(batch_size=batch_size,
                     resi_name_channel=resi_name_channel)
    n_channels = 4 if resi_name_channel else 3
    model = PointNet(channels=n_channels)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", "mse"])
    model.fit_generator(
        generator=tp.train(),
        steps_per_epoch=tp.train_steps,
        epochs=10
    )
    print(model.summary())

if __name__ == "__main__":
    train()
