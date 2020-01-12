import os
from argparse import ArgumentParser
import pickle as pk
from functools import partial

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA


class PointNet(Model):

    def __init__(self, channels, classes=4, drop_rate=0.5, weight_decay=0.01):
        super(PointNet, self).__init__()
        conv_kernel = (1, channels)
        self.conv2d_input = Conv2D(1024, conv_kernel, activation="relu")
        self.batch_norm_1 = BatchNormalization(axis=1, scale=False)
        self.batch_norm_2 = BatchNormalization()
        # self.batch_norm_3 = BatchNormalization()
        # self.batch_norm_4 = BatchNormalization()
        # self.batch_norm_5 = BatchNormalization()
        self.conv2d_64_64_1 = Conv2D(
            1024, (1, 1024), activation="relu")
        self.conv2d_64_64_2 = Conv2D(
            1024, (1, 1024), activation="relu")
        self.conv2d_64_128 = Conv2D(
            2048, (1, 1024), activation="relu")
        self.conv2d_128_1024 = Conv2D(
            2048, (1, 2048), activation="relu")
        self.maxpool = MaxPool2D((1, 2048))
        self.flatten = Flatten()
        self.dense_512 = Dense(
            512, activation="relu", kernel_regularizer=l2(weight_decay))
        self.dense_256 = Dense(
            256, activation="relu", kernel_regularizer=l2(weight_decay))
        self.dropout = Dropout(drop_rate)
        if classes == 1:
            self.dense_output = Dense(classes, activation="sigmoid")
        else:
            self.dense_output = Dense(classes, activation="softmax")
        self.perm = (0, 1, 3, 2)

    def call(self, x, training=True):
        # x = self.batch_norm_1(x, training=training)
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


class PointnetArgParser(ArgumentParser):

    def __init__(self):
        super(PointnetArgParser, self).__init__()
        self.add_argument("-i", "--input-path",
                          help="Path to the input dataset.")
        self.add_argument("-o", "--output-path",
                          help="Path for save training outputs.")
        self.add_argument("-k", "--kfold", type=int, default=10,
                          help="K-fold cross validation.")
        self.add_argument("-b", "--batch-size", type=int, default=128,
                          help="Size of mini batch.")
        self.add_argument("-e", "--epochs", default=30, type=int,
                          help="Number of training epochs.")
        self.add_argument("-d", "--dropout-rate", default=0.1, type=float,
                          help="Drop rate of the dropout layers.")
        self.add_argument("-p", "--pca", action="store_true",
                          help="If to rotate the data by aligning to"
                               " principle components.")


def scheduler(epoch):
    if epoch < 10:
        return 0.0001
    elif epoch < 50:
        return 0.00001
    else:
        return 0.000005


def key2label(key):
    if "active" in key:
        return 1
    else:
        return 0


def pad_pointcloud(pointcloud, max_len):
    new_pc = np.zeros((max_len, pointcloud.shape[1]))
    new_pc[:pointcloud.shape[0], :] = pointcloud
    return new_pc


def find_longest_pointcloud(pointclouds):
    max_len = 0
    for pc in pointclouds:
        max_len = max(pc.shape[0], max_len)
    return max_len


def load_data_kfold(k, input_path, pca_rotate=False):
    with open(input_path, "rb") as f:
        data = pk.load(f)
    X_data = list(data.values())
    #rotate points by aligning to principle components
    if pca_rotate:
        X_data = np.array(X_data)
        for i, point_cloud in enumerate(X_data):
            pca = PCA(n_components=3)
            pca.fit(point_cloud[:, 0:3])
            X_data[i][:, 0:3] = pca.transform(point_cloud[:, 0:3])
        X_data = X_data.tolist()
    # find the longest pointcloud in the dataset
    max_len = find_longest_pointcloud(X_data)
    # pad all the point clouds with 0 to max len and convert to numpy array
    pad_pointcloud_with_max_len = partial(pad_pointcloud, max_len=max_len)
    X_data = np.array(list(map(pad_pointcloud_with_max_len, X_data)))
    X_data = np.expand_dims(X_data, 3)
    # create labels besed on key
    Y_data = np.array(list(map(key2label, list(data.keys()))))
    folds = list(StratifiedKFold(n_splits=k, shuffle=True).\
        split(X_data, Y_data))
    return folds, X_data, Y_data

 
def train(in_path, out_path, k_fold, epochs, batch_size, drop_rate, pca=False):
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # load and k-fold split the training dataset
    folds, X_data, Y_data = load_data_kfold(k_fold, in_path, pca_rotate=pca)
    # print("folds: {}".format(folds))
    # print("x_data shape: {}".format(X_data.shape))
    
    # load model
    n_channels = X_data.shape[2]
    if len(Y_data.shape) == 1:
        classes = 1
    else:
        classes = Y_data.shape[-1]
    print("classes: {}".format(classes))
    model = PointNet(channels=n_channels,
                     classes=classes,
                     drop_rate=drop_rate)
    optimizer = Adam(0.0001)
    tb_callback = TensorBoard(log_dir=out_path)
    lr_callback = LearningRateScheduler(scheduler)
    auc_metric = tf.keras.metrics.AUC()
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    
    training_histories = list()
    for j, (train_idx, val_idx) in enumerate(folds):
        tf.keras.backend.clear_session()
        model = PointNet(channels=n_channels,
            classes=classes,
            drop_rate=drop_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                "binary_accuracy",
                auc_metric,
                precision_metric,
                recall_metric
            ]
        )
        # training
        # print("\nFold:", j)
        X_train_cv = X_data[train_idx]
        # print("x train cv shape: {}".format(X_train_cv.shape))
        # print("x train cv: {}".format(X_train_cv))
        Y_train_cv = Y_data[train_idx]
        sample_weights = compute_sample_weight('balanced', Y_train_cv)
        X_val_cv = X_data[val_idx]
        Y_val_cv = Y_data[val_idx]
        history = model.fit(
            x=X_train_cv,
            y=Y_train_cv,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tb_callback, lr_callback],
            sample_weight=sample_weights,
            validation_data=(X_val_cv, Y_val_cv),
            shuffle=True
        )
        training_histories.append(history)

    return training_histories


if __name__ == "__main__":
    # load required module
    import statistics as st
    # initialize arguments
    parser = PointnetArgParser()
    args = parser.parse_args()
    # train model
    histories = train(args.input_path, args.output_path, args.kfold,
                      args.epochs, args.batch_size, args.dropout_rate,
                      pca=args.pca)

    # analyze and save the training results
    val_accs = list()
    val_aucs = list()
    val_precisions = list()
    val_recalls = list()
    for his in histories:
        val_accs.append(his.history["val_binary_accuracy"])
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
    # best_val = list(map(float, best_val))
    # avg = st.mean(best_val)
    # std = st.stdev(best_val)

    # write training result to file
    os.makedirs(args.output_path, exist_ok=True)
    log_prefix = os.path.basename(args.input_path).split(".")[0]
    if args.pca:
        log_name = log_prefix+"_pointnet_training.pca.log"
    else:
        log_name = log_prefix+"_pointnet_training.log"
    with open(os.path.join(args.output_path, log_name), "w") as f:
        print("Training dataset: {}".format(log_prefix), file=f)
        print("Batch size: {}".format(args.batch_size), file=f)
        print("Epochs: {}".format(args.epochs), file=f)
        print("{}-fold cross validation.".format(args.kfold), file=f)
        print("Input: {}".format(args.input_path), file=f)
        print("Validation accuracy is {} +- {}".format(acc_avg, acc_std), file=f)
        print("AUC ROC is {} +- {}".format(auc_avg, auc_std), file=f)
        print("Precision is {} +- {}".format(
            precision_avg, precision_std), file=f)
        print("Recall is {} +- {}".format(recall_avg, recall_std), file=f)
        print("F1 score is {} +- {}".format(f1_avg, f1_std), file=f)
