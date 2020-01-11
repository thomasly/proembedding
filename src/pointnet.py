import sys
import os
import math
import logging
import argparse

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import numpy as np

from data import TOUGH_Point, TOUGH_Point_Pocket


class PointNet(Model):

    def __init__(self, channels, classes=4, drop_rate=0.5, weight_decay=0.01):
        super(PointNet, self).__init__()
        conv_kernel = (1, channels)
        self.conv2d_input = Conv2D(64, conv_kernel, activation="relu")
        self.batch_norm_1 = BatchNormalization(axis=1, scale=False)
        self.batch_norm_2 = BatchNormalization()
        # self.batch_norm_3 = BatchNormalization()
        # self.batch_norm_4 = BatchNormalization()
        # self.batch_norm_5 = BatchNormalization()
        self.conv2d_64_64_1 = Conv2D(
            64, (1, 64), activation="relu")
        self.conv2d_64_64_2 = Conv2D(
            64, (1, 64), activation="relu")
        self.conv2d_64_128 = Conv2D(
            128, (1, 64), activation="relu")
        self.conv2d_128_1024 = Conv2D(
            1024, (1, 128), activation="relu")
        self.maxpool = MaxPool2D((1, 1024))
        self.flatten = Flatten()
        self.dense_512 = Dense(
            512, activation="relu", kernel_regularizer=l2(weight_decay))
        self.dense_256 = Dense(
            256, activation="relu", kernel_regularizer=l2(weight_decay))
        self.dropout = Dropout(drop_rate)
        self.dense_output = Dense(classes, activation="softmax")
        self.perm = (0, 1, 3, 2)

    def call(self, x, training=False):
        x = self.batch_norm_1(x, training=training)
        x = self.conv2d_input(x) # batch_size x pc_len x 1 x 64
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 64 x 1
        x = self.conv2d_64_64_1(x) # batch_size x pc_len x 1 x 64
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 64 x 1
        x = self.conv2d_64_64_2(x) # batch_size x pc_len x 1 x 64
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 64 x 1
        x = self.conv2d_64_128(x) # batch_size x pc_len x 1 x 128
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 128 x 1
        x = self.conv2d_128_1024(x) # batch_size x pc_len x 1 x 1024
        x = self.batch_norm_2(x, training=training)
        x = tf.transpose(x, self.perm) # batch_size x pc_len x 1024 x 1
        x = self.maxpool(x) # batch_size x pc_len x 1 x 1
        x = self.flatten(x)
        x = self.dense_512(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense_256(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense_output(x)
        # print(x[0:5])
        return x


def parse_argv(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resi-channel",
                        help="Whether to use residue channel in the input.",
                        action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Size of mini batch.")
    parser.add_argument("-a", "--atom-channel",
                        help="Whether to use atom channel in the input.",
                        action="store_true")
    parser.add_argument("-e", "--epoch", default=10, type=int,
                        help="Number of training epochs")
    parser.add_argument("-c", "--classes", default=4, type=int,
                        help="Number of classes (default = 4)")
    parser.add_argument("-s", "--subset", default=None,
                        help="Subset used in training. Default is the whole \
                            TOUGH-C1 dataset. Options: nucleotide, heme")
    parser.add_argument("-d", "--dropout-rate", default=0.1, type=float,
                        help="Drop rate of the dropout layers during training")
    parser.add_argument("-t", "--train-test-ratio", default=0.7, type=float,
                        help="The ratio of training and testing sets.")
    parser.add_argument("-n", "--random-seed", default=0, type=int,
                        help="The random seed for splitting training and \
                            testing datasets")
    return parser.parse_args(argv)


def scheduler(epoch):
    if epoch < 3:
        return 0.001
    elif epoch < 15:
        return 0.0001
    else:
        return 0.00005
 

def _determine_channels(resi_channel, atom_channel):
    if not (resi_channel or atom_channel):
        return 3
    elif resi_channel and atom_channel:
        return 5
    else:
        return 4

def train(args):
    # training args
    resi_name_channel = args.resi_channel
    atom_name_channel = args.atom_channel
    epochs = args.epoch
    batch_size = args.batch_size
    if args.classes == 1 or args.classes == 2:
        classes = 2
    else:
        classes = args.classes
    subset = args.subset
    dropout_rate = args.dropout_rate
    train_test_ratio = args.train_test_ratio
    random_seed = args.random_seed
    # dataset
    tp = TOUGH_Point_Pocket(batch_size=batch_size,
                     resi_name_channel=resi_name_channel,
                     atom_name_channel=atom_name_channel,
                     subset=subset, label_len=classes,
                     train_test_ratio=train_test_ratio,
                     random_seed=random_seed) 
    # load model
    n_channels = _determine_channels(resi_name_channel, atom_name_channel)
    model = PointNet(channels=n_channels,
                     classes=classes,
                     drop_rate=dropout_rate)
    optimizer = Adam(0.001)
    tb_callback = TensorBoard()
    lr_callback = LearningRateScheduler(scheduler)
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    loss = "categorical_crossentropy"
    accuracy = "categorical_accuracy"
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            accuracy,
            "mse",
            auc_metric,
            precision_metric,
            recall_metric
        ]
    )
    # training
    history = model.fit_generator(
        generator=tp.train(),
        steps_per_epoch=tp.train_steps,
        epochs=epochs,
        validation_data=tp.test(),
        validation_steps=tp.test_steps,
        callbacks=[tb_callback, lr_callback]
    )
    # print(model.summary())
    return history


if __name__ == "__main__":
    import statistics as st
    from tqdm import tqdm
    for s in tqdm([
        "touch-c1", 
        "nucleotide", 
        "heme"]):
        training_histories = list()
        for n in tqdm(range(10)):
            tf.keras.backend.clear_session()
            if s == "touch-c1":
                argv = "-r -e 30 -t 0.9 -c 4 -n {}".format(n)
            else:
                argv = "-r -e 30 -t 0.9 -c 2 -s {} -n {}".format(s, n)
            args = parse_argv(argv.split())
            his = train(args)
            training_histories.append(his)
        
        # analyze and save the training results
        val_accs = list()
        val_aucs = list()
        val_precisions = list()
        val_recalls = list()
        for his in training_histories:
            val_accs.append(his.history["val_categorical_accuracy"])
            val_aucs.append(his.history["val_auc"])
            val_precisions.append(his.history["val_precision"])
            val_recalls.append(his.history["val_recall"])

        # find best epoch
        val_accs = np.array(val_accs)
        avgs = np.mean(val_accs, axis=0)
        best_epoch = np.argmax(avgs)
        # get the accuracy and standard deviation of the best epoch
        max_accs = val_accs[:, best_epoch]
        acc_avg = np.mean(max_accs)
        acc_std = np.std(max_accs)
        # get the auc score of the best epoch
        max_aucs = np.array(val_aucs)[:, best_epoch]
        auc_avg = np.mean(max_aucs)
        auc_std = np.std(max_accs)
        # get the precision, racall, f1 score of the best epoch
        max_precisions = np.array(val_precisions)[:, best_epoch]
        precision_avg = np.mean(max_precisions)
        precision_std = np.std(max_precisions)
        max_recalls = np.array(val_recalls)[:, best_epoch]
        recall_avg = np.mean(max_recalls)
        recall_std = np.std(max_recalls)
        max_f1s = 2 * max_precisions * max_recalls / (max_precisions + max_recalls)
        f1_avg = np.mean(max_f1s)
        f1_std = np.std(max_f1s)

        with open(os.path.join("training_logs", s+".log"), "w") as f:
            print("pointnet dataset: {}".format(s), file=f)
            print("residue name channel: {}".format(args.resi_channel), file=f)
            print("atom name channel: {}".format(args.atom_channel), file=f)
            print("batch size: {}".format(args.batch_size), file=f)
            print("learning rate: 0.001 with decay", file=f)
            print("epochs: {}".format(args.epoch), file=f)
            print("validation folds: 10", file=f)
            print(
                "10-fold cross validation performs the best "
                "at epoch {}".format(best_epoch),
                file=f)
            print("Accuracy is {} +- {}".format(acc_avg, acc_std), file=f)
            print("AUC ROC is {} +- {}".format(auc_avg, auc_std), file=f)
            print("Precision is {} +- {}".format(
                precision_avg, precision_std), file=f)
            print("Recall is {} +- {}".format(recall_avg, recall_std), file=f)
            print("F1 score is {} +- {}".format(f1_avg, f1_std), file=f)
