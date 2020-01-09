import os
import sys
import argparse
import pickle as pk
import random as rd
import itertools
import time

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution3D
from tensorflow.keras.layers import MaxPooling3D, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

# For reproductivity
seed = 12306
np.random.seed(seed)

class DeepDrug3DBuilder(object):
    """DeepDrug3D network
    Ouput: a model takes a 5D tensor as input and outpus probabilities of the 
    pocket binds to ATP and Heme.
    """
    @staticmethod
    def build(classes=1):
        model = Sequential()
        # Conv layer 1
        model.add(Convolution3D(
            # input_shape = (33, 33, 33, 1),
            filters=128,
            kernel_size=5,
            padding='valid',    # Padding method
            data_format='channels_first'
        ))
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 1
        model.add(Dropout(0.2))
        # Conv layer 2
        model.add(Convolution3D(
            filters=128,
            kernel_size=3,
            padding='valid',      # Padding method
            data_format='channels_first'
        ))
        model.add(LeakyReLU(alpha = 0.1))
        # Maxpooling 1
        model.add(MaxPooling3D(
            pool_size=(2,2,2),
            strides=None,
            padding='valid',     # Padding method
            data_format='channels_first'
        ))
        # Dropout 2
        model.add(Dropout(0.4))
        # FC 1
        model.add(Flatten())
        model.add(Dense(128)) # TODO changed to 64 for the CAM
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 3
        model.add(Dropout(0.4))
        # Fully connected layer 2 to shape (1) for 2 classes
        model.add(Dense(classes))
        if classes == 1:
            model.add(Activation('sigmoid'))
        else:
            model.add(Activation('softmax'))
        return model

def grid_augmentation(grid):
    rotated_grids = list()
    for fold in range(1, 4):
        # for axes in itertools.combinations(range(3), 2):
        new_grid = np.zeros(grid.shape)
        for i in range(grid.shape[0]):
            new_grid[i] = np.rot90(grid[0], fold, axes=(1, 2))
        rotated_grids.append(new_grid)
    return rotated_grids

def split_training_set(x, y, ratio, data_aug=False, shuffle=False):
    if shuffle:
        temp = list(zip(x, y))
        rd.shuffle(temp)
        x, y = zip(*temp)
    cut = int(len(x) * ratio)
    train_x, val_x = x[:cut], x[cut:]
    train_y, val_y = y[:cut], y[cut:]
    if data_aug:
        aug_x, aug_y = list(), list()
        for x, y in zip(train_x, train_y):
            aug_x.append(x)
            aug_y.append(y)
            rotated_x = grid_augmentation(x)
            aug_x += rotated_x
            aug_y += [y] * len(rotated_x)
        temp = list(zip(aug_x, aug_y))
        rd.shuffle(temp)
        train_x, train_y = zip(*temp)
    train_x = np.stack(train_x, axis=0)
    train_y = np.stack(train_y, axis=0)
    val_x = np.stack(val_x, axis=0)
    val_y = np.stack(val_y, axis=0)
    return train_x, train_y, val_x, val_y

def split_dataset_with_kfold(x, y, k, k_fold):
    d_len = len(x)
    chunk_len = int(d_len/k_fold)
    chunk_start = chunk_len * k
    chunk_end = chunk_len * (k + 1)
    train_x = x[:chunk_start] + x[chunk_end:]
    train_y = y[:chunk_start] + y[chunk_end:]
    val_x = x[chunk_start:chunk_end]
    val_y = y[chunk_start:chunk_end]
    train_x = np.stack(train_x, axis=0)
    train_y = np.stack(train_y, axis=0)
    val_x = np.stack(val_x, axis=0)
    val_y = np.stack(val_y, axis=0)
    return train_x, train_y, val_x, val_y

def train_deepdrug(batch_size,
                   lr, epoch,
                   output, k_fold=10,
                   subset="all",
                   suffix="resigrids"):
    # optimize gpu memory usage 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # load the data
    with open("../data/tough_c1/control-pocket."+suffix, "rb") as f:
        grids = list(pk.load(f).values())
    labels = [np.array([0])] * len(grids)
    if subset == "nucleotide":
        with open("../data/tough_c1/nucleotide-pocket."+suffix, "rb") as f:
            grids += list(pk.load(f).values())
        labels += [np.array([1])] * (len(grids) - len(labels))
        classes = 1
    elif subset == "heme":
        with open("../data/tough_c1/heme-pocket."+suffix, "rb") as f:
            grids += list(pk.load(f).values())
        labels += [np.array([1])] * (len(grids) - len(labels))
        classes = 1
    elif subset == "all":
        with open("../data/tough_c1/nucleotide-pocket."+suffix, "rb") as f:
            grids += list(pk.load(f).values())
        labels += [np.array([1])] * (len(grids) - len(labels))
        with open("../data/tough_c1/heme-pocket."+suffix, "rb") as f:
            grids += list(pk.load(f).values())
        labels += [np.array([2])] * (len(grids) - len(labels))
        with open("../data/tough_c1/steroid-pocket."+suffix, "rb") as f:
            grids += list(pk.load(f).values())
        labels += [np.array([3])] * (len(grids) - len(labels))
        classes = 4
    else:
        print("Invalid subset name. Please choose from all/nucleotide/heme.")
        return
    # shuffle input
    idx_list = list(range(len(labels)))
    rd.shuffle(idx_list)
    grids = list(np.array(grids)[idx_list])
    labels = list(np.array(labels)[idx_list])
    # print("grids type:", type(grids))
    # print("grid type:", type(grids[0]), "shape:", grids[0].shape)
    # print("labels type:", type(labels))
    # print("label type:", type(labels[0]), "shape:", labels[0].shape)

    histories = list()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    for k in range(k_fold):
        tf.keras.backend.clear_session()
        # get training data with respect to the kth fold
        x, y, val_x, val_y = split_dataset_with_kfold(grids, labels, k, k_fold)
        
        if classes > 1:
            y = to_categorical(y, num_classes=classes)
            val_y = to_categorical(val_y, num_classes=classes)

        # to balance different classes
        sample_weights = compute_sample_weight('balanced', y)
          
        val_data = (val_x, val_y)
        # build & compile model
        mdl = DeepDrug3DBuilder.build(classes=classes)
        adam = Adam(
            lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
            decay=0.0, amsgrad=False)
        loss = "binary_crossentropy" if classes==1 \
            else "categorical_crossentropy"
        metric = "binary_accuracy" if classes == 1 else "categorical_accuracy"
        auc_metric = tf.metrics.AUC()
        precision_metric = tf.metrics.Precision()
        recall_metric = tf.metrics.Recall()
        mdl.compile(
            optimizer=adam, loss=loss,
            metrics=[metric, auc_metric, precision_metric, recall_metric])
        # callback function for model checking
        log_dir = os.path.join('training_logs', timestamp, f"fold_{k}")
        os.makedirs(log_dir, exist_ok=True)
        tfCallBack = callbacks.TensorBoard(log_dir=log_dir)
        history = mdl.fit(
            x, y, epochs = epoch, batch_size = batch_size, 
            sample_weight=sample_weights, validation_data=val_data,
            shuffle = True, callbacks = [tfCallBack])
        histories.append(history)
    
    # log the metrics after each cross-validation training
    val_accs = list()
    val_aucs = list()
    val_precisions = list()
    val_recalls = list()
    for his in histories:
        try:
            val_accs.append(his.history["val_binary_accuracy"])
        except KeyError:
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
    # print and save the training results
    print(
        "{}-fold cross validation performs the best "
        "at epoch {}".format(k_fold, best_epoch+1))
    print("Accuracy is {} +- {}".format(acc_avg, acc_std))
    print("AUC ROC is {} +- {}".format(auc_avg, auc_std))
    print("Precision is {} +- {}".format(precision_avg, precision_std))
    print("Recall is {} +- {}".format(recall_avg, recall_std))
    print("F1 score is {} +- {}".format(f1_avg, f1_std))
    print() 
    with open(os.path.join("training_logs", timestamp, "readme"), "w") as f:
        print("dataset: {}".format(subset), file=f)
        print("grid type: {}".format(suffix), file=f)
        print("batch size: {}".format(batch_size), file=f)
        print("learning rate: {}".format(lr), file=f)
        print("epochs: {}".format(epoch), file=f)
        print("validation folds: {}".format(k_fold), file=f)
        print(
            "{}-fold cross validation performs the best "
            "at epoch {}".format(k_fold, best_epoch+1),
            file=f)
        print("Accuracy is {} +- {}".format(acc_avg, acc_std), file=f)
        print("AUC ROC is {} +- {}".format(auc_avg, auc_std), file=f)
        print("Precision is {} +- {}".format(
            precision_avg, precision_std), file=f)
        print("Recall is {} +- {}".format(recall_avg, recall_std), file=f)
        print("F1 score is {} +- {}".format(f1_avg, f1_std), file=f)
    #     # save the model
    # if output == None:
    #     mdl.save('deepdrug3d.h5')
    # else:
    #     mdl.save(output)


def myargs():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument('--bs', required = True, type=int, help = 'batch size')
    parser.add_argument('--lr', required = True, type=float, help = 
                        'initial learning rate')
    parser.add_argument('--epoch', required = True, type=int, help = 
                        'number of epochs for taining')
    parser.add_argument('--output', required = False, help = 
                        'location for the model to be saved')
    parser.add_argument('--fold', type=int, default=10, help=
                        'number of folds for k-fold cross validation')
    parser.add_argument('--set', default="all")
    parser.add_argument('--suffix', default='resigrids')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = myargs()
    train_deepdrug(args.bs, args.lr, args.epoch, args.output,
                   k_fold=args.fold, subset=args.set, suffix=args.suffix)
